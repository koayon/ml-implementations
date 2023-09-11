# Experiments inspired by `One Wide FeedForward Layer Is All You Need` (Apple Paper)

Experiments:

- Confirm results on TinyStories with a small-ish model
  - Compare original regular dense network with shared parameter version with wide shared parameter version
- New Experiments:
  - Share the attention heads as well and compare.
  - See the trade-off using MoE for the shared wide layer.
  - Try using different routers to each MoE layer.
  - Try sharing parameters in the MoE with Group Layers to get a better memory footprint to performance trade-off.
  - Try with many more attention heads with shared params across layers but routing to only a few of them. (Maybe with larger attention heads to make up for using only a few of them?)
