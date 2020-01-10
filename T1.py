import csv
import collections
import math
import random
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


# Reads in the data set
def read_data(id):
    # Actual data
    data_set = []
    # Choose the corresponding file
    file = "occupancy_" + id + ".csv"
    # Open file and reads in data
    with open(file) as data:
        reader = csv.reader(data)
        for i, row in enumerate(reader):
            # Ignore header line
            if i == 0:
                catagories = list(map(str, row))
                continue
            # Store data as float point number
            row = list(map(float, row))
            row[-1] = str(int(row[-1]))
            data_set.append(row)
    return (catagories, data_set)


# Returns the entropy of the given proba- bility distribution
def entropy(dist):
    result = 0
    total = sum(dist.values())
    for outcome in dist:
        pci = dist[outcome] / total
        if pci != 0:
            result = result - pci * math.log2(pci)
    return result


# Iterates through all the features and all possible split points of each
# feature and returns one feature and one split point for the feature that
# maximizes the expected information gain
def choose_feature_split(data):
    # Count the occupation of all data
    num_features = len(data[0])
    negative = 0
    positive = 0
    for i in range(len(data)):
        if data[i][num_features - 1] == '0':
            negative += 1
        else:
            positive += 1
    total = collections.Counter({'1': positive, '0': negative})
    total_outcomes = sum(total.values())
    # Entropy before testing
    information_gain = entropy(total)
    max_gain = -1
    split_value = -1
    which_feature = -1
    # Sort different columns and check all possible gaps
    for i in range(num_features - 1):
        # Sort data in increase order based the ith column
        data = sorted(data, key=lambda entry: entry[i])
        # Initial pair stores amount of pi and ni before or after point
        # Starts from the top of data set
        before = collections.Counter({'1': 0, '0': 0})
        after = collections.Counter({'1': positive, '0': negative})
        # Check every gaps between values to see whether it can be split point
        # Init Lx as the first data, next gap is not possible
        Lx_value = data[0][i]
        Lx_occupied = str(data[0][num_features - 1])
        possible = False
        before.update(Lx_occupied)
        after.subtract(Lx_occupied)
        for j in range(len(data) - 1):
            Ly_value = data[j + 1][i]
            Ly_occupied = str(data[j + 1][num_features - 1])
            # Next data has the same value
            if Lx_value == Ly_value:
                # Next gap is possible a split point or
                # Next data has the same result update counters and check next
                if possible or Lx_occupied == Ly_occupied:
                    before.update(Ly_occupied)
                    after.subtract(Ly_occupied)
                    # Different result, next gap is possibly a split point
                else:
                    possible = True
                    before.update(Ly_occupied)
                    after.subtract(Ly_occupied)
            # Next data has different value
            else:
                # It is a split point since left side has different outcomes
                if possible:
                    # Calculate information gain of this split point
                    gain = information_gain - sum(after.values()) / sum(total.values()) * entropy(after) - sum(
                        before.values()) / sum(total.values()) * entropy(before)
                    # This split point is better, recored it
                    if gain > max_gain:
                        split_value = (Lx_value + Ly_value) / 2
                        max_gain = gain
                        which_feature = i
                    # This split point as good as old one, MAYBE take this
                    elif gain == max_gain:
                        if bool(random.getrandbits(1)):
                            split_value = (Lx_value + Ly_value) / 2
                            max_gain = gain
                            which_feature = i
                    # Reset everything
                    Lx_value = Ly_value
                    Lx_occupied = Ly_occupied
                    before.update(Ly_occupied)
                    after.subtract(Ly_occupied)
                    possible = False
                # Left is single outcome data, need to check right
                else:
                    # Loop through right data with the same value and count
                    count = 0
                    has_different = False
                    while True:
                        # Check if there is out of data
                        if (j + 1 + count) >= len(data):
                            break
                        else:
                            temp = data[j + 1 + count]
                        # Next value is the same
                        if temp[i] == Ly_value:
                            # Same outcome
                            if str(temp[num_features - 1]) == Lx_occupied:
                                count += 1
                            # Different outcome
                            else:
                                count += 1
                                has_different = True
                        # Next value is different, stop tracking
                        else:
                            break
                    # Different outcome exists, a split point
                    if has_different:
                        # Calculate information gain of this split point
                        gain = information_gain - sum(after.values()) / sum(total.values()) * entropy(after) - sum(
                            before.values()) / sum(total.values()) * entropy(before)
                        # This split point is better, recored it
                        if gain > max_gain:
                            split_value = (Lx_value + Ly_value) / 2
                            max_gain = gain
                            which_feature = i
                        # This split point as good as old one, MAYBE take this
                        elif gain == max_gain:
                            if bool(random.getrandbits(1)):
                                split_value = (Lx_value + Ly_value) / 2
                                max_gain = gain
                                which_feature = i
                    # Reset everything
                    Lx_value = Ly_value
                    Lx_occupied = Ly_occupied
                    before.update(Ly_occupied)
                    after.subtract(Ly_occupied)
                    possible = False
    return (which_feature, split_value)


index = 1


# DataNode class based on Node, with more fileds and functions
class DataNode(Node):
    ''' Field: feature stores the feature of this node
               value stores the split value of this node
               where stroes it's the left 
    '''
    feature = -1
    value = -1
    label = ""

    # stores useful initial information in the node
    def init_node(self, feature, value):
        self.feature = feature
        self.value = value
        return

    # produces the node based on the data points left and recursively 
    # produces the children of the current node if necessary
    def split_node(self, data, max_depth=None):
        # Check all data and put into two sets
        left = []
        right = []
        # Trace the number of positive data on both sides 
        count_left_positive = 0
        count_right_positive = 0
        global index
        for i in range(len(data)):
            # Data belongs to left side
            if data[i][self.feature] < self.value:
                left.append(data[i])
                if data[i][-1] == '1':
                    count_left_positive += 1
            # Data belongs to right side
            else:
                right.append(data[i])
                if data[i][-1] == '1':
                    count_right_positive += 1
        # No example left           
        if len(left) == 0:
            if count_right_positive < (len(data) / 2):
                left_node = DataNode("No" + " ID=" + str(index))
                index += 1
            else:
                left_node = DataNode("Yes" + " ID=" + str(index))
                index += 1
        # Left side are all negative create a leaf
        elif count_left_positive == 0:
            left_node = DataNode("No" + " ID=" + str(index))
            index += 1
        # Left side are all positive create a leaf    
        elif count_left_positive == len(left):
            left_node = DataNode("Yes" + " ID=" + str(index))
            index += 1
        # Detect feature and split value for left
        else:
            which_feature_l, split_value_l = choose_feature_split(left)
            # No features left
            if which_feature_l == -1:
                if count_left_positive < (len(left) / 2):
                    left_node = DataNode("No" + " ID=" + str(index))
                    index += 1
                else:
                    left_node = DataNode("Yes" + " ID=" + str(index))
                    index += 1
            else:
                left_node = DataNode(catagories[which_feature_l] + " ID=" + str(index))
                index += 1
                left_node.init_node(which_feature_l, split_value_l)
                left_node.split_node(left)

        # No example left           
        if len(right) == 0:
            if count_left_positive < (len(data) / 2):
                right_node = DataNode("No" + " ID=" + str(index))
                index += 1
            else:
                right_node = DataNode("Yes" + " ID=" + str(index))
                index += 1
        # Right side are all negative create a leaf
        elif count_right_positive == 0:
            right_node = DataNode("No" + " ID=" + str(index))
            index += 1
        # Right side are all positive create a leaf
        elif count_right_positive == len(right):
            right_node = DataNode("Yes" + " ID=" + str(index))
            index += 1
        # Detect feature and split value for right
        else:
            which_feature_r, split_value_r = choose_feature_split(right)
            # No features left
            if which_feature_r == -1:
                if count_right_positive < (len(right) / 2):
                    right_node = DataNode("No" + " ID=" + str(index))
                    index += 1
                else:
                    right_node = DataNode("Yes" + " ID=" + str(index))
                    index += 1
            else:
                right_node = DataNode(catagories[which_feature_r] + " ID=" + str(index))
                index += 1
                right_node.init_node(which_feature_r, split_value_r)
                right_node.split_node(right)

        # Update the inheritor
        left_node.parent = self
        left_node.label = "&lt;" + str(self.value)
        right_node.parent = self
        right_node.label = "&gt;" + str(self.value)
        return

    # Returns the decision/label given a data point
    # If the given node is a leaf node, this function returns the decision
    # in the leaf node. Otherwise, this function recursively calls itself
    # on the child nodes until it reaches a decision
    def get_decision(self, one_data_point):
        if self.feature == -1:
            return self.name[0]
        else:
            if one_data_point[self.feature] < self.value:
                return self.children[0].get_decision(one_data_point)
            else:
                return self.children[1].get_decision(one_data_point)


# Tree class based on RenderTree, with more fields and functions
class MyTree(RenderTree):
    ''' Fields: node stores the root node of the tree
    '''

    # Initializes the decision tree by creating the root node
    def init_tree(self, data):
        # Find the split point for root node
        feature, value = choose_feature_split(data)
        # Create root node
        root = DataNode(catagories[feature] + " ID=0")
        root.init_node(feature, value)
        self.node = root
        return

    # Trains the decision tree using the provided data and up to the
    # maximum depth if specified. In our implementation, this function 
    # calls the split_node function to produce the root node of the 
    # decision tree
    def train_tree(self, data, max_depth=None):
        # Split root node by using data
        self.node.split_node(data)
        return self.node.height + 1

    # Returns the prediction accuracy of the decision tree on the provided 
    # data. In our implementation, this function calls the get_decision 
    # function on the root node of the tree
    def get_prediction_accuracy(self, data):
        # Test accuracy
        correct = 0
        for item in data:
            result = self.node.get_decision(item)
            if result == "Y":
                result = "1"
            else:
                result = "0"
            if result == item[-1]:
                correct += 1
        return float(correct / len(data))

    # Produces a plot of the tree and saves the plot in the file at the 
    # provided path
    def plot_tree(self, file_path):
        # for pre, fill, node in self:
        #    print("%s%s" % (pre, node.name))  
        DotExporter(self.node, edgeattrfunc=lambda p, c: "label=<" + c.label + ">").to_picture("t1.pdf")
        return


# Main function
if __name__ == "__main__":
    # Which data set to use
    id = "A"
    # Reads in the data set
    catagories, data = read_data(id)
    train_data = data
    test_data = data
    # Create a tree
    tree = MyTree("")
    # Init the tree with root node
    tree.init_tree(train_data)
    # Train the tree with data set
    print("Depth of the tree: " + str(tree.train_tree(train_data)))
    # Print the prediction accuracy
    print("Prediction accuracy: " + str(tree.get_prediction_accuracy(test_data)))
    # Plot the tree
    print("\nT1 generated")
    tree.plot_tree("")
