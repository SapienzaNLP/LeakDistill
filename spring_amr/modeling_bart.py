from transformers.models.bart.modeling_bart import *
from transformers.models.bart.modeling_bart import _expand_mask
from spring_amr.adapter import *
import spring_amr.utils as utils


class BartEncoderWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, adapter_config=None):
        super().__init__(config, embed_tokens)

        self.adapter_config = adapter_config
        device = self.device

        if adapter_config.mlp_mode:
            mlp_params = self.adapter_config.mlp_params if hasattr(self.adapter_config, 'mlp_params') else dict()
            self.adapters = nn.ModuleList([AdpaterWrapper(
                SimpleAdapter(config.d_model, **mlp_params).to(device)
            ).to(device) for _ in range(len(self.adapter_config.mlp_layers))])
            self.adapt_dict = {k: self.adapters[i] for i, k in
                                     enumerate(self.adapter_config.mlp_layers)}

        graph_class = MultiGraphConvAdapter if self.adapter_config.graph_type == 'multi' else GraphConvAdapter
        self.graph_adapters = nn.ModuleList([graph_class(config.d_model, config.d_model, **self.adapter_config.graph_params)
                                             for _ in range(len(self.adapter_config.graph_layers))])
        self.graph_adapt_dict = {k: self.graph_adapters[i] for i, k in enumerate(self.adapter_config.graph_layers)}
        # This flag controls the leaked mode. True value means structural adapters are switched on
        self.leak_path = False
        # This attribute contains WAG data when the leaked mode is on
        self.orig_graph_data = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        #leak_path=False
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos

        is_full_graph_mode = hasattr(self, 'orig_graph_data') and self.orig_graph_data is not None and self.orig_graph_data[1] is not None

        if (self.adapter_config.graph_mode or self.leak_path) and is_full_graph_mode:
            # Full Graph mode
            orig_tokens_mask, extra_tokens_mask, new_attention_mask, extra_states_mask = self.orig_graph_data[1]
            padding_mask = (new_attention_mask == False)
            extra_states = self.calc_extra_node_states(self.orig_graph_data[0], extra_states_mask)
            padding_states = self.embed_tokens(torch.ones((padding_mask.sum(),), dtype=int).to(hidden_states.device))

            if self.adapter_config.extra_nodes_as_input:
                hidden_states = inputs_embeds
                attention_mask = new_attention_mask
                extra_states *= self.embed_scale
                new_hidden_states = self.combine_extra_states(orig_tokens_mask, extra_tokens_mask, padding_mask,
                                          hidden_states[input_ids != self.padding_idx], extra_states, padding_states)
                embed_pos = self.embed_positions(new_hidden_states.shape)
                hidden_states = new_hidden_states + embed_pos

        bool_attention_mask = attention_mask.bool()

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                graph_info = {}
                graph_info['x'] = hidden_states
                graph_info['input_ids'] = input_ids

                # MLP Adapters
                if self.adapter_config.mlp_mode and idx in self.adapt_dict:
                    x = graph_info['x']
                    x[bool_attention_mask] = self.adapt_dict[idx](bool_attention_mask, x)
                    hidden_states = x

                # Structural Adapters
                if (self.adapter_config.graph_mode or self.leak_path) and idx in self.graph_adapt_dict:
                    cur_edges = self.cur_edges if self.leak_path or (
                            self.adapter_config.graph_mode and self.adapter_config.leak_mode) else None

                    if not self.adapter_config.extra_nodes_as_input and is_full_graph_mode:
                        graph_hidden_states = self.combine_extra_states(orig_tokens_mask, extra_tokens_mask, padding_mask,
                                                                      hidden_states[bool_attention_mask],
                                                                      extra_states, padding_states)
                        graph_hidden_states, _, _ = self.graph_adapt_dict[idx](
                            graph_hidden_states,
                            mask=torch.logical_or(orig_tokens_mask, extra_tokens_mask),
                            edges=cur_edges,
                        )
                        hidden_states[bool_attention_mask] = graph_hidden_states[orig_tokens_mask]
                        extra_states = graph_hidden_states[extra_tokens_mask]
                    else:
                        hidden_states, _, _ = self.graph_adapt_dict[idx](
                            hidden_states,
                            mask=bool_attention_mask,
                            edges=cur_edges,
                        )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions,
        )

    def calc_extra_node_states(self, token_ids, seq_mask):
        token_embeds = self.embed_tokens(token_ids)
        pad_mask = (token_ids != 1).int() # Here 1 is from our dataset not bart padding_idx
        diviser = pad_mask.sum(dim=-1).unsqueeze(-1)
        diviser[seq_mask == False] = 1
        token_embeds = (token_embeds * pad_mask.unsqueeze(3)).sum(dim=-2) / diviser
        return token_embeds[seq_mask]

    def combine_extra_states(self, orig_tokens_mask, extra_tokens_mask, padding_mask, orig_states, extra_states, padding_states):
        res_states = torch.zeros((orig_tokens_mask.shape[0], orig_tokens_mask.shape[1], orig_states.shape[-1]),
                                 dtype=orig_states.dtype, device=orig_states.device)
        res_states[orig_tokens_mask] = orig_states
        res_states[extra_tokens_mask] = extra_states
        res_states[padding_mask] = padding_states
        return res_states


class BartStructAdaptModel(BartModel):
    def __init__(self, config: BartConfig):
        BartPretrainedModel.__init__(self, config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoderWithAdapter(config, self.shared, utils.dict_to_class(config.adapter_configs['encoder']))
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


class BartStructAdaptForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartStructAdaptModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class AMRBartForConditionalGeneration(BartStructAdaptForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self._rev = None

    def init_reverse_model(self):
        rev = AMRBartForConditionalGeneration(self.model.config)
        rev.model.shared = self.model.shared
        rev.model.encoder = self.model.encoder
        rev.model.decoder.embed_tokens = self.model.decoder.embed_tokens
        rev.model.decoder.embed_positions = self.model.decoder.embed_positions
        self._rev = rev

    @property
    def rev(self):
        if self._rev is None:
            return self

        return self._rev

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        #align_graph_edges=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.nll_loss(
                lm_logits.log_softmax(-1).contiguous().view(-1, lm_logits.size(-1)),
                labels.contiguous().view(-1),
                ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        output = Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        return output
