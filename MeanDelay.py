#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:42:50 2018

@author: jshree
"""
import os
import json

def find_delay(animation_files):
    list_delay = []
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)
            index = 0
            while index < len(animation):
                delay = 0
                prev_data = animation[index-1]
                curr_data = animation[index]
                if index == 0:
                    prev_data = curr_data
                    index += 1
                    continue
                if change_label(prev_data[0], curr_data[0]):
                    if not same_agents(prev_data[0], curr_data[0]):  
                        if agents_moving(prev_data, curr_data):
                            index += 1
                            continue
                        while not agents_moving(prev_data, curr_data):
                            delay += 1
                            if index+1 >= len(animation):
                                break
                            else:
                                next_data = animation[index+1] 
                                if change_label(curr_data[0], next_data[0]):
                                    index += 1
                                    continue
                            prev_data = curr_data
                            curr_data = next_data
                            index += 1
                        if delay > 1:
                            list_delay.append(delay)
                index +=  1

    meandelay = sum(list_delay)/len(list_delay)
    return(meandelay)
            
def change_label(prevLabel, currLabel):
    return True if prevLabel != currLabel else False

def same_agents(prevLabel, currLabel):
    if prevLabel.count('-') != currLabel.count('-'):
        return False
    if prevLabel.count('-') == 1:
        if prevLabel.split('-')[0] == currLabel.split('-')[0]:
            return True 
    else:
        if prevLabel.split('-')[0] == currLabel.split('-')[0] and prevLabel.split('-')[-1] == currLabel.split('-')[-1]:
            return True 
    return False
    
def find_agents(currLabel):
    if currLabel.count('-') == 1:
        return [currLabel.split('-')[0]]
    else:
        return [currLabel.split('-')[0], currLabel.split('-')[-1]]
                

def agents_moving(a1, a2):
    global shapeNumber
    diff = []
    agents = find_agents(a2[0])

    agent = agents[0]
    if a1[shapeNumber[agent][0]:shapeNumber[agent][1]] != a2[shapeNumber[agent][0]:shapeNumber[agent][1]]:
        diff.append(True)
    else:
        diff.append(False)
    return any(diff)  

animation_filenames = ['example_animations/' + filename for filename in os.listdir('example_animations')]
shapeNumber = {}
shapeNumber['BT'] = [1,4]
shapeNumber['LT'] = [4,7]
shapeNumber['C'] = [7,10]
shapeNumber['D'] = [10,11]
shapeNumber['stopped'] = [1,11]

if __name__ == "__main__":
    meandelay = find_delay(animation_filenames)
    print("The mean delay is", meandelay)