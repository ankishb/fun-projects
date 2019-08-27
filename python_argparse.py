
import argparser

if __name__ == '__main__':
	parser = argparser.ArgumentParser(
		description='Demo of argparse'
	)

	# add positional variable, which will be parsed at same position/ order in which it is defined
	parser.add_argument('num1', help='Number 1', type=float) # by default, type is string
	# python file.py 800

	# add optional variable, which will be parsed in any order as long as variable is defined
	parser.add_argument('--num1', help='Number 1', type=float, dafault='100.0')
	# python file.py --num1 80

	# add optional variable, which will be parsed in any order as long as variable is defined
	parser.add_argument('-n','--num1', help='Number 1', type=float, dafault='100.0')
	# python file.py -n=80
	# python file.py --help

	# To clean help query
	parser.add_argument('-n','--num1', metavar="", help='Number 1', type=float, dafault='100.0')
	# python file.py -n=80

	# To prevent error
	parser.add_argument('-n','--num1', required=True, metavar="", help='Number 1', type=float, dafault='100.0')
	# python file.py -n=80



	# parse agruments
	args = parser.parse_args()