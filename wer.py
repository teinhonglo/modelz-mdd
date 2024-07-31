import numpy as np
import io

def edit_distance(r, h):
    '''
    This function is to calculate the edit distance of the reference sentence and the hypothesis sentence.
    The main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting the reference sentence.
        h -> the list of words produced by splitting the hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def get_step_list(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting the reference sentence.
        h -> the list of words produced by splitting the hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    '''
    x = len(r)
    y = len(h)
    step_list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]:
            step_list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1] + 1:
            step_list.append("i")
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] + 1:
            step_list.append("s")
            x = x - 1
            y = y - 1
        else:
            step_list.append("d")
            x = x - 1
    return step_list[::-1]

def aligned(step_list, r, h, result):
    '''
    This function is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        step_list -> the list of steps.
        r      -> the list of words produced by splitting the reference sentence.
        h      -> the list of words produced by splitting the hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    output = io.StringIO()
    print("REF:", end=" ", file=output)
    for i in range(len(step_list)):
        if step_list[i] == "i":
            count = sum(1 for j in range(i) if step_list[j] == "d")
            index = i - count
            print(" " * len(h[index]), end=" ", file=output)
        elif step_list[i] == "s":
            count1 = sum(1 for j in range(i) if step_list[j] == "i")
            index1 = i - count1
            count2 = sum(1 for j in range(i) if step_list[j] == "d")
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2]) - len(r[index1])), end=" ", file=output)
            else:
                print(r[index1], end=" ", file=output)
        else:
            count = sum(1 for j in range(i) if step_list[j] == "i")
            index = i - count
            print(r[index], end=" ", file=output)
    print(file=output)
    print("HYP:", end=" ", file=output)
    for i in range(len(step_list)):
        if step_list[i] == "d":
            count = sum(1 for j in range(i) if step_list[j] == "i")
            index = i - count
            print(" " * len(r[index]), end=" ", file=output)
        elif step_list[i] == "s":
            count1 = sum(1 for j in range(i) if step_list[j] == "i")
            index1 = i - count1
            count2 = sum(1 for j in range(i) if step_list[j] == "d")
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1]) - len(h[index2])), end=" ", file=output)
            else:
                print(h[index2], end=" ", file=output)
        else:
            count = sum(1 for j in range(i) if step_list[j] == "d")
            index = i - count
            print(h[index], end=" ", file=output)
    print(file=output)
    print("EVA:", end=" ", file=output)
    for i in range(len(step_list)):
        if step_list[i] == "d":
            count = sum(1 for j in range(i) if step_list[j] == "i")
            index = i - count
            print("D" + " " * (len(r[index]) - 1), end=" ", file=output)
        elif step_list[i] == "i":
            count = sum(1 for j in range(i) if step_list[j] == "d")
            index = i - count
            print("I" + " " * (len(h[index]) - 1), end=" ", file=output)
        elif step_list[i] == "s":
            count1 = sum(1 for j in range(i) if step_list[j] == "i")
            index1 = i - count1
            count2 = sum(1 for j in range(i) if step_list[j] == "d")
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1]) - 1), end=" ", file=output)
            else:
                print("S" + " " * (len(h[index2]) - 1), end=" ", file=output)
        else:
            count = sum(1 for j in range(i) if step_list[j] == "i")
            index = i - count
            print("C" + " " * (len(r[index]) - 1), end=" ", file=output)
    print(file=output)
    print("WER:", result, file=output)
    return output.getvalue()

def calc_wer(r, h):
    """
    This is a function that calculates the word error rate in ASR.
    You can use it like this: calc_wer("what is it".split(), "what is".split()) 
    """
    # Build the matrix
    d = edit_distance(r, h)

    # Find out the manipulation steps
    step_list = get_step_list(r, h, d)

    # Print the result in an aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result = f"{result:.2f}%"
    result = aligned(step_list, r, h, result)
    return result

if __name__ == "__main__":
    result = calc_wer("I love natural language processing".split(), "I like language processing a lot".split())
    print(result)

    for r in result.split("\n")[:-2]:
        print(r)
        print(r.split(), len(r.split()))