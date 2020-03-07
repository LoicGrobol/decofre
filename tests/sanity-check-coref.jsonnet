{
  'mention-detection': {
    epochs: 0,
    'train-file': 'file:/fixtures/mentions.json',
    'dev-file': 'file:/fixtures/mentions.json',
    'test-file': 'file:/fixtures/mentions.json',
  },
  'antecedent-scoring': {
    epochs: 2,
    'train-file': 'file:/fixtures/antecedents.json',
    'dev-file': 'file:/fixtures/antecedents.json',
    'test-file': 'file:/fixtures/antecedents.json',
  },
  'lexicon-source': 'file:/fixtures/mentions.json',
}
