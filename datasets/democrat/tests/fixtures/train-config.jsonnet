{
 'mention-detection': {
    epochs: 2,
    lr: '1e-3',
    'weight-decay': '0.',
    'train-file': 'file:/full/mentions/',
    'dev-file': 'file:/full/mentions/',
    'test-file': 'file:/full/mentions/',
    'lr-schedule': 'step',
  },
  'antecedent-scoring': {
    epochs: 2,
    lr: '1e-4',
    'lr-schedule': 'step',
    'score-anaphoricity': true,
    'train-file': 'file:/full/antecedents/',
    'dev-file': 'file:/full/antecedents/',
    'test-file': 'file:/full/antecedents/',
  },
  'word-embeddings': 'file:/../../local/cc.fr.300.vec',
  'lexicon-source': 'file:/full/mentions/',
}
