import day1
import day2
import day3
import day4
import day5
import day6
import day7
import day8
import day9
import day10
import day11
import day12
import day13
import day14
import day15
import day16
import day17
import day18
import day19
import day20
import day21
import day22
import day23
import day24
import day25
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--days', nargs='+', choices=[x for x in range(1, 26)], type=int)
    args = parser.parse_args()

    days_to_run = {x for x in range(1, 26)}
    if args.days is not None:
        days_to_run &= set(args.days)

    days_functions = [
        day1.day1,
        day2.day2,
        day3.day3,
        day4.day4,
        day5.day5,
        day6.day6,
        day7.day7,
        day8.day8,
        day9.day9,
        day10.day10,
        day11.day11,
        day12.day12,
        day13.day13,
        day14.day14,
        day15.day15,
        day16.day16,
        day17.day17,
        day18.day18,
        day19.day19,
        day20.day20,
        day21.day21,
        day22.day22,
        day23.day23,
        day24.day24,
        day25.day25,
    ]

    start = time.time()

    for day in days_to_run:
        days_functions[day - 1]()

    print("\nReport:\ndays = {}\nduration = {:.2f}s".format(days_to_run, time.time() - start))
