def count_special(string):
    count = 0
    for char in string:
        if not(char.islower()) and not(char.isupper()) and not(char.isdigit()):
            if char != ' ':
                count += 1
    return count
# print (count_special('91943reskf bskfb 32rbc9 zvch dfvkdfbv\/.>;?/:{]'))