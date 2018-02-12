"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode

# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    list_ = text
    dict_ = {}
    for item in list_:
        if item not in dict_:
            dict_[item] = 1
        else:
            dict_[item] += 1
    return dict_


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    list_ = order_dict(freq_dict)
    list_.reverse()
    node = HuffmanNode(None, HuffmanNode(None), HuffmanNode(None))
    if len(list_) == 1:
        x = list_.pop()
        node = HuffmanNode(None, HuffmanNode(x[1]), HuffmanNode(None))
    while len(list_) > 1:
        x = list_.pop()
        y = list_.pop()
        node = assign_node(x[1], y[1])
        tup = (x[0] + y[0], node)
        list_.insert(0, tup)
        i = 0
        if len(list_) > 1:
            while not i + 2 > len(list_) and list_[i][0] < list_[i + 1][0]:
                list_[i], list_[i + 1] = list_[i + 1], list_[i]
                i = i+1
    return node


def assign_node(left, right):
    """ Assigns a HuffmanNode based on the types of left and right.

    @type left: int|str|HuffmanNode
    @type right: int|str|HuffmanNode
    @rtype: HuffmanNode

    >>> assign_node(1, 2)
    HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None))
    >>> node1 = HuffmanNode(None, 1, HuffmanNode(2, None, None))
    >>> node2 = 2
    >>> node3 = assign_node(node1, node2)
    >>> node4 = HuffmanNode(None, HuffmanNode(None, 1, \
    HuffmanNode(2, None, None)), HuffmanNode(2, None, None))
    >>> node3 == node4
    True
    """
    if type(left) is HuffmanNode and type(right) is HuffmanNode:
        return HuffmanNode(None, left, right)
    elif type(left) is HuffmanNode and type(right) is not HuffmanNode:
        return HuffmanNode(None, left, HuffmanNode(right))
    elif type(left) is not HuffmanNode and type(right) is HuffmanNode:
        return HuffmanNode(None, HuffmanNode(left), right)
    else:
        return HuffmanNode(None, HuffmanNode(left), HuffmanNode(right))


def order_dict(dict_):
    """ Return a list of tuples (frequency, item) ordered by frequency of items.

    @type dict_: dict
    @rtype: list[(int, int)]
    >>> d = {65: 1, 66: 2, 67: 1}
    >>> order_dict(d)
    [(1, 65), (1, 67), (2, 66)]
    """
    ordered_keys = []
    ordered_values = list(dict_.values())
    ordered_values.sort()
    keys = list(dict_.keys())
    for value in ordered_values:
        for key in keys:
            if dict_[key] == value:
                tup = (value, key)
                ordered_keys.append(tup)
                keys.remove(key)
    return ordered_keys


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    codes = {}
    leaves = get_leaves(tree)
    for item in leaves:
        codes[item] = get_code(tree, item)
    return codes


def get_leaves(tree):
    """ Return a list of leaves in the tree rooted at HuffmanNode.

    @type tree: HuffmanNode
    @rtype: list

    >>> left1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> left = HuffmanNode(None, left1, right1)
    >>> left2 = HuffmanNode(None, HuffmanNode(4), HuffmanNode(7))
    >>> right2 = HuffmanNode(None, HuffmanNode(12), HuffmanNode(1))
    >>> right = HuffmanNode(None, left2, right2)
    >>> tree = HuffmanNode(None, left, right)
    >>> get_leaves(tree)
    [3, 2, 9, 10, 4, 7, 12, 1]
    """
    if tree.is_leaf():
        return [tree.symbol]
    elif tree.left is not None and tree.right is not None:
        return gather_lists([get_leaves(tree.left), get_leaves(tree.right)])
    else:
        return []


def gather_lists(list_):
    """
    Concatenate all the sublists of L and return the result.

    @param list list_: list of lists to concatenate
    @rtype: list[object]

    >>> gather_lists([[1, 2], [3, 4, 5]])
    [1, 2, 3, 4, 5]
    >>> gather_lists([[6, 7], [8], [9, 10, 11]])
    [6, 7, 8, 9, 10, 11]
    """
    new_list = []
    for l in list_:
        new_list += l
    return new_list


def get_code(tree, value):
    """ Return the code for value in tree rooted at HuffmanNode.

    @type tree: HuffmanNode
    @type value: int | str
    @rtype: str

    >>> left1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right1 = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> left = HuffmanNode(None, left1, right1)
    >>> left2 = HuffmanNode(None, HuffmanNode(4), HuffmanNode(7))
    >>> right2 = HuffmanNode(None, HuffmanNode(12), HuffmanNode(1))
    >>> right = HuffmanNode(None, left2, right2)
    >>> tree = HuffmanNode(None, left, right)
    >>> value = 10
    >>> get_code(tree, value)
    '011'
    """
    if tree.left is None and tree.right is None and tree.symbol == value:
        return ''
    elif tree.left is None and tree.right is None and tree.symbol != value:
        return 'x'
    else:
        left = '0' + get_code(tree.left, value)
        right = '1' + get_code(tree.right, value)
        return right if 'x' in left else left


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    n = 0
    while tree.number is None:
        number_helper(tree, n)
        n = n + 1


def number_helper(tree, n):
    """ Traverses through the tree in preorder and sets the number of the first
    open internal node to n.

    @type tree: HuffmanNode
    @type n: int
    @rtype: None
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_helper(tree, 0)
    >>> tree.left.number
    0
    """
    if tree.is_leaf():
        tree.number = None
    elif tree.left.is_leaf() and tree.right.is_leaf():
        tree.number = n
    elif tree.left.number is None and not tree.left.is_leaf():
        number_helper(tree.left, n)
    elif tree.right.number is None and not tree.right.is_leaf():
        number_helper(tree.right, n)
    else:
        tree.number = n


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)
    freqs = freq_dict
    chars = list(codes.keys())
    occurences = []
    total = list(freq_dict.values())
    for item in chars:
        if item is not None:
            occurences.append(len(codes[item])*freqs[item])
    return sum(occurences)/sum(total)


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bytes_ = []
    bits = []
    for byte in list(text):
        k = codes[byte]
        for char in k:
            bytes_.append(char)
    bytes_.reverse()
    while len(bytes_) > 0:
        j = ''
        while len(j) < 8:
            if len(bytes_) != 0:
                j = j + bytes_.pop()
            else:
                j = j + '0'
        bits.append(bits_to_byte(j))
    return bytes(bits)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if not tree.is_leaf():
        result = [tree_to_bytes(tree.left) if tree.left is not None else []] + \
                 [tree_to_bytes(tree.right) if tree.right is not None else []]
        result.append(get_data(tree))
        return bytes(gather_lists(result))
    else:
        return bytes([])


def get_data(node):
    """ Return the ReadNode data in a list for the given node.

    @type node: HuffmanNode
    @rtype: list

    >>> node = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> get_data(node)
    [0, 3, 0, 2]
    """
    if not node.is_leaf():
        data = []
        if node.left.is_leaf():
            data.append(0)
            data.append(node.left.symbol)
        else:
            data.append(1)
            data.append(node.left.number)
        if node.right.is_leaf():
            data.append(0)
            data.append(node.right.symbol)
        else:
            data.append(1)
            data.append(node.right.number)
        return data
    return []


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    k = node_lst[root_index]
    if k.l_type == 0 and k.r_type == 0:
        return HuffmanNode(None, HuffmanNode(k.l_data),
                           HuffmanNode(k.r_data))
    elif k.l_type == 0 and k.r_type == 1:
        return HuffmanNode(None, HuffmanNode(k.l_data),
                           generate_tree_general(node_lst, k.r_data))
    elif k.l_type == 1 and k.r_type == 0:
        return HuffmanNode(None, generate_tree_general(node_lst, k.l_data),
                           HuffmanNode(k.r_data))
    elif k.l_type == 1 and k.r_type == 1:
        return HuffmanNode(None, generate_tree_general(node_lst, k.l_data),
                           generate_tree_general(node_lst, k.r_data))


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    hnode_list = postgen_helper(node_lst, root_index)
    hnode_list.reverse()
    node = HuffmanNode()
    while len(hnode_list) > 1:
        if len(hnode_list) == 1:
            x = hnode_list.pop()
            node = HuffmanNode(None, x)
        else:
            x = hnode_list.pop()
            y = hnode_list.pop()
            node = HuffmanNode(None, x, y)
            hnode_list.insert(0, node)
    return node


def postgen_helper(node_lst, root_index):
    """ Returns a list of HuffmanNodes with leaf children from the list of
    ReadNodes in node_lst.

    @type node_lst: list[ReadNode]
    @type root_index: int
    @rtype: list[HuffmanNode]
    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> postgen_helper(lst, 2)
    [HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None))]
    """
    list_ = []
    for item in node_lst[:root_index + 1]:
        if item.l_type == 0 and item.r_type == 0:
            node = HuffmanNode(None, HuffmanNode(item.l_data),
                               HuffmanNode(item.r_data))
            list_.append(node)
        elif item.l_type == 0 and item.r_type == 1:
            node = HuffmanNode(item.l_data)
            list_.append(node)
        elif item.l_type == 1 and item.r_type == 0:
            node = HuffmanNode(item.r_data)
            list_.append(node)
    return list_


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    result = []
    cts_dict = gen_cts_dict(tree)
    bits = ''.join([byte_to_bits(byte) for byte in text])
    x = 0
    y = size
    z = 1
    while x < len(bits) and y > 0:
        while bits[x:z] not in list(cts_dict.keys()):
            z = z + 1
        result.append(cts_dict[bits[x:z]])
        x = z
        y = y - 1
    return bytes(result)


def gen_cts_dict(tree):
    """ Return a code to symbol dictionary from Huffman Tree rooted at
    HuffmanNode tree

    @type tree: HuffmanNode
    @rtype: dict

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result = gen_cts_dict(t)
    >>> r1 = {'1': 2, '0': 3}
    >>> r2 = {'0': 3, '1': 2}
    >>> result == r1 or result == r2
    True
    """
    cts_dict = {}
    symbols = get_leaves(tree)
    for symbol in symbols:
        code = get_code(tree, symbol)
        cts_dict[code] = symbol
    return cts_dict


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))

# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    leaf_list = order_dict2(freq_dict)
    optimize(tree, leaf_list)


def order_dict2(dict_):
    """ Return a list of keys ordered by frequency of items

    @type dict_: dict
    @rtype: list

    >>> d = {65: 1, 66: 2, 67: 1}
    >>> order_dict2(d)
    [65, 67, 66]
    """
    ordered_keys = []
    ordered_values = list(dict_.values())
    ordered_values.sort()
    keys = list(dict_.keys())
    for value in ordered_values:
        for key in keys:
            if dict_[key] == value:
                ordered_keys.append(key)
                keys.remove(key)
    return ordered_keys


def optimize(node, leaf_list):
    """ Optimizes the tree rooted at HuffmanNode node by replacing its values
    with a list of leaf symbols ordered in frequency.

    @type node: HuffmanNode
    @type leaf_list: list
    @rtype: None

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> leaf_list = order_dict2(freq)
    >>> optimize(tree, leaf_list)
    >>> avg_length(tree, freq)
    2.31
    """
    if node.is_leaf():
        node.symbol = leaf_list.pop()
    else:
        optimize(node.left, leaf_list)
        optimize(node.right, leaf_list)


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")

    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
