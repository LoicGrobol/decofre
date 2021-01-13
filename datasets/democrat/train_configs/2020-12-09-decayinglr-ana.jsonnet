{
  'mention-detection': {
    epochs: 50,
    patience: 5,
    lr: '1e-3',
    'weight-decay': '0.',
    'train-file': 'file:/../local/processed/train/mentions/',
    'dev-file': 'file:/../local/processed/dev/mentions/',
    'test-file': 'file:/../local/processed/test/mentions/',
    'lr-schedule': 'step',
  },
  'antecedent-scoring': {
    epochs: 50,
    patience: 5,
    lr: '1e-4',
    'lr-schedule': 'step',
    'score-anaphoricity': true,
    'train-file': 'file:/../local/processed/train/antecedents/',
    'dev-file': 'file:/../local/processed/dev/antecedents/',
    'test-file': 'file:/../local/processed/test/antecedents/',
  },
  'word-embeddings': 'file:/../local/cc.fr.300.vec',
  'lexicon-source': 'file:/../local/processed/train/mentions/',
}
