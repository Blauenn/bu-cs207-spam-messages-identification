# Pure code

import sys
import numpy

import matplotlib.pyplot as plt

import sklearn
import random
import statistics

import csv
import string

colors_ham_spam = ["#ff7dfb", "#ffcf69"]

# Opening CSV
with open("spam.csv", "r", encoding="cp1252") as file:
  reader = csv.reader(file)
  data = list(reader)

#####################
# Data cleanup
#####################
joined_data = [["v1", "v2"]]

for row in data[1:]:
  message_parts = row[1:]
  
  # Remove column with empty value before joining
  while message_parts and message_parts[-1] == "":
    message_parts.pop()
  
  row_data = []
  row_data.append(row[0])
  row_data.append(",".join(message_parts))
  
  joined_data.append(row_data)

### Sanity check ###

# Total rows and column
print("Total rows:", len(joined_data))
print("Columns per row:", len(joined_data[1]))

empty_labels = 0			# labels : ham / spam
empty_messages = 0

for row in joined_data[1:]:
  if row[0] == "":
    empty_labels += 1
    
  if row[1] == "":
    empty_messages += 1

# Label check (Make sure that there's only "ham" and "spam")
labels = set()

for row in joined_data[1:]:
    labels.add(row[0])


#####################
# Data separation
#####################

headers = [joined_data[0]]
ham = []
spam = []

for row in joined_data[1:]:
  if row[0] == "ham":
    ham.append(row)
  elif row[0] == "spam":
    spam.append(row)
    
labels = ["Ham", "Spam"]
counts = [len(ham), len(spam)]
all_messages = sum(counts)

#####################
# Identifying features
#####################

ham_word_counts = []
spam_word_counts = []

### Mean, median, mode

# Mean

ham_mean = sum(ham_word_counts) / len(ham)
spam_mean = sum(spam_word_counts) / len(spam)

for row in ham:
  message = row[1]
  words = message.split()
  ham_word_counts.append(len(words))
  
for row in spam:
  message = row[1]
  words = message.split()
  spam_word_counts.append(len(words))
  
ham_mean = sum(ham_word_counts) / len(ham)
spam_mean = sum(spam_word_counts) / len(spam)

print("\nMean:")
print("Ham mean:", round(ham_mean, 2))
print("Spam mean:", round(spam_mean, 2))

# Median

ham_sorted = sorted(ham_word_counts)
spam_sorted = sorted(spam_word_counts)

ham_middle = len(ham_sorted) // 2
spam_middle = len(spam_sorted) // 2

ham_median = (
	ham_sorted[ham_middle - 1] + ham_sorted[ham_middle]
) / 2

spam_median = (
	spam_sorted[spam_middle - 1] + spam_sorted[spam_middle]
) / 2

print("\nMedian:")
print("Ham median:", ham_median)
print("Spam median:", spam_median)

# Mode

ham_word_frequencies = {}
spam_word_frequencies = {}

# Go through all numbers of word counts, add to a dict.
for word_count in ham_word_counts:
  if word_count not in ham_word_frequencies:
    ham_word_frequencies[word_count] = 1
  else:
    ham_word_frequencies[word_count] += 1

# Find mode by sorting and then pick the first value
sorted_ham_frequencies = sorted(
  ham_word_frequencies.items(),
  key=lambda item: item[1],
  reverse=True
)

for word_count in spam_word_counts:
    if word_count not in spam_word_frequencies:
        spam_word_frequencies[word_count] = 1
    else:
        spam_word_frequencies[word_count] += 1

sorted_spam_frequencies = sorted(
    spam_word_frequencies.items(),
    key=lambda item: item[1],
    reverse=True
)

ham_mode = sorted_ham_frequencies[0][0]
spam_mode = sorted_spam_frequencies[0][0]

print("\nMode:")
print("Ham mode:", ham_mode)
print("Spam mode:", spam_mode)

#####################
# Variance
#####################

ham_squared_differences = []
spam_squared_differences = []

for word_count in ham_word_counts:
  difference = word_count - ham_mean
  squared_differences = difference ** 2
  ham_squared_differences.append(squared_differences)

for word_count in spam_word_counts:
  difference = word_count - spam_mean
  squared_differences = difference ** 2
  spam_squared_differences.append(squared_differences)
  
ham_variance = sum(ham_squared_differences) / len(ham_squared_differences)
spam_variance = sum(spam_squared_differences) / len(spam_squared_differences)

print("\nVariance:")
print("Ham variance:", ham_variance)
print("Spam variance:", spam_variance)

#####################
# Standard deviation
#####################

ham_standard_deviation = ham_variance ** 0.5
spam_standard_deviation = spam_variance ** 0.5

print("\nStandard deviation:")
print("Ham SD:", ham_standard_deviation)
print("Spam SD:", spam_standard_deviation)

#####################
# Bayes' Theorem
#####################

# Probabilities of ham and spam

total_messages = len(ham) + len(spam)

ham_probability = len(ham) / total_messages
spam_probability = len(spam) / total_messages

print("\nProbabilities...")
print("of ham:", ham_probability)
print("of spam:", spam_probability)

# Count all the words in messages

# Words to skip
stop_words = {
    "a", "an", "the", "this", "that",
    "to", "and", "in", "is", "am", "are",
    "i", "you", "u", "my", "im", "me", "of",
    "for", "it", "have", "but", "your", "so",
    "on", "at", "if", "do",
    "be", "we", "with", "ill",
    "when", "how", "up", "ur", "what",
    "got", ".", "or", "was",
    "its", "then", "?", "he", "out",
    "ltgt"
}

translator = str.maketrans("", "", string.punctuation)

ham_word_frequencies = {}
spam_word_frequencies = {}

# Go through all messages, add words to the dictionary
# Skip numbers
for row in ham:
  message = row[1].lower().split()
  
  for word in message:
    # Remove punctuation from words
    word = word.translate(translator)
    
    if not word or word.isdigit() or word in stop_words:
      continue
      
    ham_word_frequencies[word] = (
      ham_word_frequencies.get(word, 0) + 1
    )

for row in spam:
  message = row[1].lower().split()

  for word in message:
    word = word.translate(translator)
    
    if not word or word.isdigit() or word in stop_words:
      continue
    
    spam_word_frequencies[word] = (
      spam_word_frequencies.get(word, 0) + 1
    )
    
# Sort the most frequent words

sorted_ham_words = sorted(
  ham_word_frequencies.items(), 
  key=lambda item: item[1], 
  reverse=True
)
sorted_spam_words = sorted(
  spam_word_frequencies.items(), 
  key=lambda item: item[1], 
  reverse=True
)

print("\nTop 10 Ham Words::")
for word, count in sorted_ham_words[:10]:
  print(f"{word}: {count}")

print("\nTop 10 Spam Words::")
for word, count in sorted_spam_words[:10]:
  print(f"{word}: {count}")


### Comparing words

def compare_word_frequency(word):
  ham_count = ham_word_frequencies.get(word, 0)
  spam_count = spam_word_frequencies.get(word, 0)
  difference = ham_count - spam_count
  
  return ham_count, spam_count, difference

#####################
# CHANGE WORD TO COMPARE HERE!!!
#####################
word_to_compare = str(input("\nEnter word to compare: "))

compared_word = compare_word_frequency(word_to_compare.lower())

compared_word_ham = compared_word[0]
compared_word_spam = compared_word[1]
difference = abs(compared_word[2])

print("\nComparing:", word_to_compare)
print("Ham:", compared_word_ham, "words.")
print("Spam:", compared_word_spam, "words.")
if compared_word[2] > 0:
  print("\nHam has", difference, "more occurrences.")
elif compared_word[2] < 0:
  print("\nSpam has", difference, "more occurrences.")
else:
  print("\nBoth have the same number of occurrences.")
  

# Bayes' Theorem in action

total_compared_word = compared_word_ham + compared_word_spam
spam_given_word = compared_word_spam / total_compared_word

print(f"\nP(spam | {word_to_compare}): {spam_given_word:.2%}")

###

word_given_spam = compared_word_spam / len(spam)

print(f"P({word_to_compare} | spam): {word_given_spam:.2%}")

###

print(f"P(spam): {spam_probability:.2%}")

###

word_probability = total_compared_word / all_messages

print(f"P({word_to_compare}): {word_probability:.2%}")

### Full Bayes' calculation

spam_given_word_bayes = (
	word_given_spam * spam_probability
) / word_probability

print(f"P(spam | {word_to_compare}): {spam_given_word_bayes:.2%}")