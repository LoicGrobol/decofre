{
  'mention-detection': {
    epochs: 50,
    patience: 5,
    lr: '1e-3',
    'weight-decay': '0.',
    'train-file': 'file:/../data/train/mentions/',
    'dev-file': 'file:/../data/dev/mentions/',
    'test-file': 'file:/../data/test/mentions/',
    'lr-schedule': 'step',
  },
  'antecedent-scoring': {
    epochs: 50,
    patience: 5,
    lr: '1e-4',
    'train-file': 'file:/../data/train/antecedents/',
    'dev-file': 'file:/../data/dev/antecedents/',
    'test-file': 'file:/../data/test/antecedents/',
  },
  'word-embeddings': 'file:/../data/fasttext-d300-neg10-OSCAR_fr.vec',
  'lexicon-source': 'file:/../data/train/mentions/',
}
