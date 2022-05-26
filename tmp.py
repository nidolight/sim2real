
def solution(new_id):
    #1
    answer = new_id.lower()

    #2
    characters = "~!@#$%^&*()=+[{]}:?,<>/"
    answer = ''.join( x for x in answer if x not in characters) 

    #3
    tmp = answer[0]
    for i in range(1, len(answer)):
        if answer[i] == '.' and answer[i-1] == '.':
            pass
        else:
            tmp += answer[i]
    answer = tmp

    #4
    if answer[0] == '.':
        if len(answer) == 1:
            answer = ''
        answer = answer[1:]
    if len(answer) != 0 and answer[-1] == '.':
        answer = answer[:-1]

    #5
    if answer == '':
        answer = 'a'
        
    #6
    if len(answer) > 15:
        answer = answer[:15]
        if answer[-1] == '.':
            answer = answer[:-1]
    
    #7
    while(len(answer) < 3):
        answer += answer[-1]

    return answer

input_str = input()
print(solution(input_str))
