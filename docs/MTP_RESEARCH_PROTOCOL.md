# MTP research protocol

## Do not rank MTP by its raw auxiliary loss

At random initialization a 32,768-way token predictor has a uniform cross-entropy of `ln(32768) ≈ 10.397`. Predicting t+2/t+3 is harder than next-token prediction, so a large early MTP loss is expected. The useful question is whether the auxiliary task improves the **main model** after training enough for the MTP module to learn.

The old synthetic profiler used random token IDs and random labels. Its MTP loss can measure numerical stability, memory and runtime overhead, but **cannot measure MTP usefulness**.

## vNext2 comparisons

On the selected proxy architecture and LR, start every variant from scratch on the same real contiguous text and fixed token budget:

- no MTP;
- Aster low-rank MTP depth 1;
- DeepSeek-style sequential MTP depth 1.

Rank by:

1. main validation loss (`eval_main_loss`) at fixed model tokens;
2. wall-clock/tokens-per-second cost;
3. MTP auxiliary trajectory as a diagnostic, not the objective;
4. later, speculative acceptance/latency after a production incremental drafter exists.

Never compare total loss across MTP0 and MTP>0; total loss includes the auxiliary term by construction.

## DeepSeek-style alignment

For source hidden state `h_i`, the one-step MTP module receives the **actual embedding of token `x_{i+1}`**, combines normalized embedding and normalized hidden through a 2d→d projection, runs a full Aster block, final-normalizes, and predicts `x_{i+2}` through the shared output head. Document-boundary masks are propagated so MTP cannot learn an arbitrary transition across packed documents.

Aster vNext2 intentionally supports one sequential MTP layer only. The production speculative path is a separate problem: during inference the future token is a proposal, not a known teacher token, and the MTP layer needs incremental state/cache management. vNext2 refuses to present full-prefix re-verification as a speed optimization.

## Evidence already available

The two-seed, real-data 8,388,608-token vNext2 comparison found that low-rank
MTP-1 improved mean final **main-model** validation loss from approximately
`5.9590` to `5.9434`. Median training throughput fell from approximately
`15.29k` to `12.54k` tokens/s, an 18% tax. This is evidence that joint MTP can
help the backbone at fixed tokens, but not evidence that it wins at equal wall
time or that the current reference decoder is faster.

The DeepSeek-style sequential arms failed before producing a training step due
to the old Transformer Engine FP8 shape integration. They are integration
failures, not negative quality evidence.

## Final-scale gate

Choose the total-parameter-matched dense/MoE body first. Then compare that body
with and without jointly trained low-rank MTP-1 at the same seed, data order and
token budget. MTP-1 adds 984,320 parameters to either 1,280-wide body, so the
dense/MoE total-parameter match is preserved when both carry it.

Promotion requires all of the following:

1. improved or acceptably unchanged main-model validation quality at equal
   tokens and equal wall time;
2. finite gradients, exact resume and practical laptop fit;
3. exact greedy output parity;
4. a cached incremental verifier that improves measured accepted tokens/s.

The MTP weights must be learned jointly (or in an explicit later distillation
phase). Merely changing `mtp_depth` after backbone pretraining produces an
untrained drafter and is not a supported deployment path.
