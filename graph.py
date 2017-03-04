class Node(object):
  def __init__(self, value):
    super(Node, self).__init__()
    self.value = value
    self.parents = []
    self.children = []

  def add_child(self, child):
    node = Node(child)
    node.parents.append(self)
    self.children.append(node)
    return node

  def join(self, other, n):
    """Join leaves of the graph if their n-2 ancestors and themselves are identical"""
    if len(self.children) != 0:
      raise Exception("Cannot join nodes that aren't leaves!")
    if self.value != other.value:
      return False
    # We only need to check one parent since a node can only have multiple parents after joining
    self_check = self
    other_check = other
    for i in range(n-2):
      if self_check.parents[0].value == other_check.parents[0].value:
        self_check = self_check.parents[0]
        other_check = other_check.parents[0]
      else:
        return False
    # We passed the check so we join
    self.parents.extend(other.parents)
    for parent in other.parents:
      parent.children.remove(other)
      parent.children.append(self)
    other.parents = []
    return True

  def generate_ngrams(self, n):
    """Traverse the graph towards the root, generating all possible ngrams"""
    if n == 1:
      return [[self.value]]
    else:
      return [ngram + [self.value] for parent in self.parents for ngram in parent.generate_ngrams(n-1)]

  def generate_first_ngram(self, n):
    if n == 1:
      return [self.value]
    else:
      return [*[ngram for ngram in self.parents[0].generate_first_ngram(n-1)], self.value]

  def __str__(self):
    return self.value

  def __repr__(self):
    return self.value
