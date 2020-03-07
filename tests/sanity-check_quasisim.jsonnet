{
  'mention-detection': {
    epochs: 2,
    'train-file': 'file:/../datasets/ancor/data/fixtures-tiny/mentions.json',
    'dev-file': 'file:/../datasets/ancor/data/fixtures-tiny/mentions.json',
    'test-file': 'file:/../datasets/ancor/data/fixtures-tiny/mentions.json',
  },
  'antecedent-scoring': {
    epochs: 2,
    'train-file': 'file:/../datasets/ancor/data/fixtures-tiny/antecedents.json',
    'dev-file': 'file:/../datasets/ancor/data/fixtures-tiny/antecedents.json',
    'test-file': 'file:/../datasets/ancor/data/fixtures-tiny/antecedents.json',
  },
  'lexicon-source': 'file:/../datasets/ancor/data/fixtures-tiny/mentions.json',
  'training-scheme': {
    type: 'quasisimultaneous',
    steps: {
      antecedent: 1,
      detection: 3,
    },
  },
}
