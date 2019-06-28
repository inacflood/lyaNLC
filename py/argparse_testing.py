#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:51:54 2019

@author: iflood
"""

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Add 2 numbers')
    
    parser.add_argument('--out-dir',type=str,default=None,required=True,
        help='Output directory')
    
    parser.add_argument('--first_num',type=float,default=0.0,required=True,
        help='First number to add')
    
    parser.add_argument('--second_num',type=float,default=0.0,required=True,
        help='Second number to add')
    
    args = parser.parse_args()
    
    sumd = args.first_num + args.second_num
    
    print(sumd)