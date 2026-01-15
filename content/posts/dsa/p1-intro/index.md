+++
date = '2026-01-05T15:41:17+05:30'
draft = false
title = 'DSA: Part 1 - Dynamic Programming'
series = ['dsa']
series_order = 1
+++

Dynamic programming always struck me as an incredibly contrived solution to certain problems. 
Often, I would unsuspectingly try to reason about some problem and build some sort of recursive/brute-force approach, concluding that this *must be* the optimal solution, only to be hit with TLE on any non-trivial testcases. 
Upon looking at the solutions, I would be met again with the familiar yet indecipherable `dp` tables, until I just resolved to buildd a **formal understanding** of what DP entails and how to apply it myself.

## DAGs
One thing that would have helped me *immensely* would have been an understanding of a substructure common to **every problem** where DP is applicable - the DAG (*Directed Acyclic Graph*).
Once I understood what this amalgamation of technical jargon *actually meant*, I was able to sniff out DP solutions much more succesfully.

The 3 parts each tell you something about what it is:
1. **directed** - one vertex may connect to another vertex but not the other way round. In other words, the edge `A -> B` existing doesn't imply the existence of `B -> A`.
2. **acyclic** - a cycle is any path on a graph that takes you from the start back to the start, and every vertex *other than the start* is visited a maximum of one time.
3. **graph** - [a data structure](https://web.cecs.pdx.edu/~sheard/course/Cs163/Doc/Graphs.html) which represents objects (vertices/nodes) and their relationships (edges/connections).






