{
  'mention-detection': {
    epochs: 100,
    patience: 5,
    'train-file': 'file:/../datasets/ancor/data/fixtures/mentions/',
    'dev-file': 'file:/../datasets/ancor/data/fixtures/mentions/',
    'test-file': 'file:/../datasets/ancor/data/fixtures/mentions/',
    'lr-schedule': 'step',
  },
  'antecedent-scoring': {
    epochs: 100,
    patience: 5,
    'train-file': 'file:/../datasets/ancor/data/fixtures/antecedents/',
    'dev-file': 'file:/../datasets/ancor/data/fixtures/antecedents/',
    'test-file': 'file:/../datasets/ancor/data/fixtures/antecedents/',
  },
  'word-embeddings': 'file:/../datasets/ancor/data/cc.fr.300.vec',
  'lexicon-source': 'file:/../datasets/ancor/data/fixtures/mentions/',
}
