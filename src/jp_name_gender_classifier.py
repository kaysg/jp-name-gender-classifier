# coding: utf-8

import glob
import unicodedata
import string
import random
import time
import math
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---- Define Functions and Classes ----
def readLines(filename):
    """
    Read a file and store the lines into a list
    :param filename: single filename
    :return: list of lines in the file
    """
    lines = codecs.open(filename, 'r', 'utf-8').read().strip().split('\n')
    return lines


def line_to_tensor(line):
    """
    Convert a line to a tensor (line_len x 1 x 57)
    :param line: line in string format
    :return: tensor in one hot vector
    >>> line
    >>> 'ツバサ'
    >>> tensor
    >>> (0 ,.,.) =
    >>> Columns 0 to 18
    >>>     1   0   0   0 ..
    >>> (1 ,.,.) =
    >>> Columns 0 to 18
    >>>     0   1   0   0 ..
    >>> (2 ,.,.) =
    >>> Columns 0 to 18
    >>>     0   0   1   0 ..
    >>> ...
    >>> Columns 38 to 84
    >>>     ... 0   0   0   0
    >>> [torch.FloatTensor of size 3x1x85]
    """

    tensor = torch.zeros(len(line), 1, n_letters)  # n_letters=85
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


def category_from_output(output):
    """
    Convert tensor output from NN model to cateory data
    :param output: output from NN model
    :return: category data
    >>> output
    >>> -0.2106 -1.6611
    >>> [torch.FloatTensor of size 1x2]
    >>>
    >>> top_n
    >>> -0.2106 [torch.FloatTensor of size 1x1] #=> -0.2106 > -1.6611
    >>> top_i
    >>> 0 [torch.LongTensor of size 1x1]
    >>> category_i
    >>> 0 [int]
    >>> all_categories[category_i]
    >>> 'boys_name'
    """
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def train(category_tensor, line_tensor):
    """
    Perform training
    :param category_tensor: label tensor for input
    :param line_tensor: line tensor for input
    :return: NN model output, loss
    """
    lstm.zero_grad()
    hidden = lstm.init_hidden()

    # Calculate loss
    for i in range(line_tensor.size()[0]):
        output, hidden = lstm(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)  # nn.NLLLoss()

    # Calculate gradient
    loss.backward()

    # Performs a single optimization step including update of model parameters
    optimizer.step()

    return output, loss.data[0]


def time_since(since):
    """
    Calculate processed time in min and sec
    :param since: time data when processing started
    :return: string to show min and sec
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def evaluate(line_tensor):
    """
    Perform evaluation
    :param line_tensor: input tensor for evaluation
    :return: output from NN model
    """
    hidden = lstm.init_hidden()

    # Get prediction for all samples/lines
    for i in range(line_tensor.size()[0]):
        output, hidden = lstm(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=2):
    """
    Perform prediction
    :param input_line: input tensor for prediction
    :param n_predictions: number of data to generate from one sample
    :return: prediction result
    >>> input_line
    >>> サトシ
    >>>(-0.05) boys_name
    >>>(-2.97) girls_name
    """
    hidden = lstm.init_hidden()
    var_line = Variable(line_to_tensor(input_line))

    # Get prediction for all samples/lines
    for i in range(var_line.size()[0]):
        output, hidden = lstm(var_line[i], hidden)

    # Get top N categories (probability, category_index)
    topv, topi = output.data.topk(n_predictions, 1, True)

    # Store prediction result to a patent
    predictions = []
    print('\n> %s' % input_line)
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])


# ---- Pre-processing ----
# Parameter Initialization
category_lines = {}
all_categories = []
all_losses = []
n_hidden = 128
learning_rate = 0.005
n_epochs = 100000
print_every = 5000
plot_every = 1000
current_loss = 0

all_filenames = glob.glob('../data/jp_names/*.txt')
all_letters = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトド" \
              "ナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴー"
n_letters = len(all_letters)

# Generate category_lines (list to contain all names)
for filename in all_filenames:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
'''
>>> category_lines['girls_name']
>>> ['アイ',
>>>  'アイカナ',
>>>  'アイリ',
>>> ...]
'''
# Generate model, cost function, and optimizer function
n_categories = len(all_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)  # 85, 128, 2

criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# ---- Training ----
# Perform training
start = time.time()
for epoch in range(1, n_epochs + 1):
    # Get a random training input and target value
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))

    # Training
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (
        epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))
    # 10000 10% (0m 21s) 0.8611 カズマ / girls_name ✗ (boys_name)
    # 15000 15% (0m 31s) 0.0329 リョウマ / boys_name ✓
    # 20000 20% (0m 41s) 0.0103 ミナ / girls_name ✓
    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Plot loss
plt.figure()
plt.plot(all_losses)

# ---- Confusion Matrix ----
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    # Get a random training input and target value
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))

    # Evaluation
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Plot Confusion Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

# ---- Test ----
# Test using sample data
predict('サトシ')
predict('クミコ')
predict('アキヒコ')

# > サトシ
# (-0.05) boys_name
# (-2.97) girls_name
#
# > クミコ
# (-0.00) girls_name
# (-8.81) boys_name
#
# > アキヒコ
# (-0.12) boys_name
# (-2.20) girls_name

# Get performance data
y_true, y_pred = [], []
for category in all_categories:
    for line in category_lines[category]:
        line_tensor = Variable(line_to_tensor(line))
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        y_true.append(category_i)
        y_pred.append(guess_i)

report = classification_report(y_true, y_pred)
print(report)
print("Accuracy: %s" % accuracy_score(y_true, y_pred))

#              precision    recall  f1-score   support
#
#           0       0.93      0.97      0.95       234
#           1       0.97      0.93      0.95       222
#
# avg / total       0.95      0.95      0.95       456
#
# Accuracy: 0.951754385965

# [Reference]
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# https://github.com/napsternxg/pytorch-practice/blob/master/utils.py
