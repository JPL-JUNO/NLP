"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@Time: 2023-04-22 13:15:07
"""
# Python中有两个“官方”的正则表达式包，这里使用的是re包，因
# 为它安装在所有版本的Python中。而regex包只在较新版本的Python中安装，
# 与re相比，regex的功能要强大很多

import re

# '|'表示“OR”，'*'表示前面的字
# 符在出现0次或多次的情况下都可以匹配。因此，这里的正则表达式将匹配
# 以“hi” “hello”或“hey”开头、后面跟着任意数量的空格字符再加上任意数量字母的问候语

r = '(hi|hello|hey)[ ]*([a-z]*)'
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)
re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
re.match(r, "hey, what's up?", flags=re.IGNORECASE)

r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
# 可以编译正则表达式，这样就不必在每次使用它们时指定选项（或标志）
re_greeting = re.compile(r, flags=re.IGNORECASE)
re_greeting.match('Hello Rosa')
re_greeting.match('Hello Rosa').groups()
re_greeting.match("Good Morning Rosa")
# 注意，这个正则表达式无法识别（匹配）录入错误
assert re_greeting.match('Good Manning Rosa') is None

# 这里的聊天机器人可以将问候语的不同部分分成不同的组，但是他不会知道Rosa是一个著名的姓
# 因为这里没有一个模式来匹配名后面的任何字符
re_greeting.match('good evening Rosa Parks').groups()

re_greeting.match("Good Morn'n Rosa")
re_greeting.match('yo Rosa')

my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot', 'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = ''
match = re_greeting.match(input())

if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print('Good one.')
    elif at_name.lower() in my_names:
        print('Hi {}, How are you?'.format(greeter_name))
