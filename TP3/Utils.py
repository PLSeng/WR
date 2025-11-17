class Utils:
    @staticmethod
    def create_vocabulary(tokens_list):
        vocabulary = set()
        for tokens in tokens_list:
            for token in tokens:
                vocabulary.add(token)
        return sorted(list(vocabulary))

    @staticmethod
    def create_bow_vector(tokens, vocabulary):
        bow_vector = [0] * len(vocabulary)
        for token in tokens:
            if token in vocabulary:
                index = vocabulary.index(token)
                bow_vector[index] += 1
        return bow_vector

    @staticmethod
    def bowvectors_to_dict(vocabulary, bow_vectors):
        col_sums = [sum(col) for col in zip(*bow_vectors)]
        return dict(zip(vocabulary, col_sums))