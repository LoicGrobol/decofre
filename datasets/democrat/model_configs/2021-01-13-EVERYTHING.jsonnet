{
  encoder: {
    type: 'bert',
    pretrained: 'camembert-base',
    combine_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    project: false,
    fine_tune: true,
    span_encoding_dim: 512,
    hidden_dim: 512,
    soft_dropout_rate: 0.2,
    hard_dropout_rate: 0.2,
    features: [
      {
        name: 'length',
        vocabulary_size: 10,
        embeddings_dim: 20,
      },
    ],
  },
  detector: {
    ffnn_dim: 256,
    dropout: 0.3,
  },
  scorer: {
    ffnn_dim: 256,
    dropout: 0.2,
    mention_new: 'from_refined',
    features: [
      {
        name: 'w_distance',
        vocabulary_size: 10,
        embeddings_dim: 20,
      },
      {
        name: 'u_distance',
        vocabulary_size: 10,
        embeddings_dim: 20,
      },
      {
        name: 'm_distance',
        vocabulary_size: 10,
        embeddings_dim: 20,
      },
      {
        name: 'spk_agreement',
        digitization: 'lexicon',
        embeddings_dim: 2,
      },
      {
        name: 'token_incl',
        vocabulary_size: 11,
        embeddings_dim: 20,
      },
      {
        name: 'token_com',
        vocabulary_size: 11,
        embeddings_dim: 20,
      },
      {
        name: 'overlap',
        digitization: 'lexicon',
        embeddings_dim: 2,
      },
    ],
  },
}
