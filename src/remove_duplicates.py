#!/usr/bin/python3
import csv

def main():

	f=open("kddcup.data","r")
	line_set={"0"}
	while True:
		line=f.readline()
		if line =="":
			break
		line=line[:-2]
		line_set.add(line+"\n")
	line_set.remove("0")
	w = open("noduplicateskddcup.csv", "a")
	for i in line_set:
		w.write(i)

if __name__ == "__main__":
	exit(main())
