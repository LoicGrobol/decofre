{
 'mention-detection': {
    epochs: 2,
    lr: '1e-3',
    'weight-decay': '0.',
    'train-file': 'file:/train/mentions/',
    'dev-file': 'file:/train/mentions/',
    'test-file': 'file:/train/mentions/',
    'lr-schedule': 'step',
  },
  'antecedent-scoring': {
    epochs: 2,
    lr: '1e-4',
    'train-file': 'file:/train/antecedents/',
    'dev-file': 'file:/train/antecedents/',
    'test-file': 'file:/train/antecedents/',
  },
  'word-embeddings': 'file:/../../local/cc.fr.300.vec',
  'lexicon-source': 'file:/train/mentions/',
}
