
# T Y P E   D E F I N G,   F O R   V A R I A B L E   &   F U N C T I O N
def pick(l: list, index: int = 0) -> int:
    if not isinstance(l, list):
        raise TypeError
    return l[index]


# M U L T I P R O C E S S I N G    M O D U L E
import multiprocessing as mp

def print_func(num1, num2):
	print("Hii! testing mp tool {} : {}".format(num1, num2))

# if __name__ == '__main__':
for i in range(100):
	p = mp.Process(target=print_func, args=(i, i+1))
	p.start()
	p.join()



# M A P    R E D U C E     F I L T E R
nums = [1,2,3,4,5,6]
evens = list(filter(lambda x: x%2, nums))
doubles = list(map(lambda x: x*2, evens))
sum = reduce(lambda a,b: a+b, evens)