{
  encoder: {
    type: 'elmo',
    span_encoding_dim: 500,
    elmo_options_file: 'file:/../../data/frelmo-1.0/elmo_fr_1.0_options.json',
    elmo_weight_file: 'file:/../../data/frelmo-1.0/elmo_fr_1.0_weights.hdf5',
    hidden_dim: 300,
    features: [
      {
        name: 'length',
        vocabulary_size: 10,
        embeddings_dim: 20,
      },
    ],
  },
  detector: {
    ffnn_dim: 200,
    dropout: 0.6,
  },
  scorer: {
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
        name: 'overlap',
        digitization: 'lexicon',
        embeddings_dim: 2,
      },
    ],
  },
}
