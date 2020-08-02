from collections import defaultdict
from random import random
import time
import sys
import numpy as np

class HmmNERTAG():
    def __init__(self,trainfolderpath):
        self.sentenceandtags = self.dataset(trainfolderpath)
        self.tags = list(self.sentenceandtags.values())
        self.sentences = list(self.sentenceandtags.keys())
        self.tag_uniques,self.tag_bigramcounts,self.tag_unigramcounts = self.Ngram(self.tags)
        self.states = self.tag_uniques
        self.initial_probs,self.emission_counts,self.transition_counts = self.HMM(self.sentenceandtags)
        self.observations, sentence_bigram_counts, sentence_unigrams = self.Ngram(self.sentences)
        self.tag_uniques.remove("start")
        self.tag_uniques.remove("end")
    def dataset(self,folderpath):
        datafile = open(folderpath, "r")
        datafile.readline()
        datafile.readline()
        sentence_tagsdict = defaultdict(lambda: [])
        sentence = ""
        tags = []
        for line in datafile.readlines():
            line = line[:-1].lower() + "\n"

            if line == "\n":
                sentence = sentence.rstrip(" ")
                sentence_tagsdict[sentence] = tags
                sentence = ""
                tags = []
                continue
            line = line.lower()
            line = line.rstrip("\n")
            line = line.split(" ")
            sentence += line[0] + " "
            tags.append(line[-1])

            datafile.close()
        return sentence_tagsdict

    def HMM(self,listofsentences):
        initial_probs = {}
        emissions = defaultdict(lambda: defaultdict(lambda: 0))
        transition_counts = defaultdict(lambda: defaultdict(lambda: 1))

        # print(emissions['ORG']["EU"])
        for k, v in self.sentenceandtags.items():
            k = k.split()
            if "start" in v:
                v.remove("start")
                v.remove("end")
            for sent_element, tag in zip(k, v):
                emissions[tag][sent_element] += 1
        # initial probs
        for tagkey in ['i-misc', 'b-misc', 'i-org', 'i-per', 'b-org', 'b-per', 'o', 'b-loc', 'i-loc']:
            if self.tag_bigramcounts[('start', tagkey)]==0:
                initial_probs[tagkey] = np.log((self.tag_bigramcounts[('start', tagkey)]+1)/(len(self.sentences)+9))
            else:
                initial_probs[tagkey] = np.log(self.tag_bigramcounts[('start', tagkey)] / len(self.sentences))
        for tag1 in ['i-misc', 'start', 'end', 'b-misc', 'i-org', 'i-per', 'b-org', 'b-per', 'o', 'b-loc', 'i-loc']:
            if tag1 == 'start':
                continue
            for tag2 in ['i-misc', 'end', 'b-misc', 'i-org', 'i-per', 'b-org', 'b-per', 'o', 'b-loc', 'i-loc']:
                transition_counts[tag1][tag2] = self.tag_bigramcounts[(tag1, tag2)]


        return (initial_probs, emissions, transition_counts)
    def Ngram(self,sentences):
        unique_words = set()
        bigramcounts = defaultdict(lambda: 0)
        unigramcounts = defaultdict(lambda: 0)
        for temp in sentences:
            sentence = temp
            if not isinstance(sentence, list):
                sentence = sentence.split(" ")
            for i in range(len(sentence)):
                unigramcounts[sentence[i]] += 1
            sentence.insert(0, "start")
            sentence.append("end")
            for i in range(len(sentence) - 1):
                bigramcounts[(sentence[i], sentence[i + 1])] += 1
                unique_words.add(sentence[i])
                unique_words.add(sentence[i + 1])
        return list(unique_words), bigramcounts, unigramcounts

    def get_emission_prob(self,test_tag, test_emission):
        ret = 0.0
        if self.emission_counts[test_tag][test_emission] == 0:
            ret = np.log(1 / (sum(list(self.emission_counts[test_tag].values())) + len(self.observations)))
        else:
            ret = np.log(self.emission_counts[test_tag][test_emission] / sum(list(self.emission_counts[test_tag].values())))
        return ret

    def get_transition_probs(self, testtag1, testtag2):
        if self.transition_counts[testtag1][testtag2] == 0:
            return np.log(1 / (self.tag_unigramcounts[testtag1] + 9))
        else:
            return np.log(self.transition_counts[testtag1][testtag2] / self.tag_unigramcounts[testtag1])
    def viterbi(self,hmm_out, testsentences):
        init_probs, emit_counts, transitions_counts = hmm_out
        test_tags=[]
        for test in testsentences:
            test_sentence = test.split(" ")
            table = np.zeros((len(self.states), len(test_sentence)))
            trace = np.full(len(test_sentence), -1)
            for state in range(len(self.states)):
                table[state][0] = init_probs[self.states[state]] + self.get_emission_prob(self.states[state],test_sentence[0])
            current_state = np.argmax(table[:, 0])
            trace[0] = current_state

            num_repeats = 0

            for sentence_part in range(1, len(test_sentence)):
                maxarg = np.argmax(table[:, sentence_part - 1])
                maxval = max(table[:, sentence_part - 1])

                for state in range(len(self.states)):
                    table[state][sentence_part] = maxval +self. get_transition_probs(self.states[maxarg],
                                                                                self.states[state]) +self.get_emission_prob(
                        self.states[state], test_sentence[sentence_part])

                next_state = np.argmax(table[:, sentence_part])
                if next_state == maxarg:
                    num_repeats += 1
                else:
                    num_repeats = 0
                trace[sentence_part] = next_state
                current_state = next_state

            traces = []
            for t in trace:
                traces.append(self.states[t])
            test_tags.append(traces)

        return test_tags
    def accuracy(self,test_tags,gold_sequences):
        trues = 0
        total = 0
        for ypred, gold_sequence in zip(test_tags, gold_sequences):
            for y, goldy in zip(ypred, gold_sequence):
                if y == goldy:
                    trues += 1
        for sent in gold_sequences:
            total += len(sent)
        return trues / total

def main() :
    trainfiledir = "train.txt"
    testfiledir = "test.txt"

    model = HmmNERTAG(trainfiledir )

    testsentence_tags = model.dataset(testfiledir)
    testsentences = list(testsentence_tags.keys())
    gold_sequences = list(testsentence_tags.values())
    hmmmodel = model.HMM(model.sentenceandtags)
    ypreds = model.viterbi(hmmmodel,testsentences)
    print(model.accuracy(ypreds,gold_sequences))
main()








