10 14
0 1
1 2
2 3
3 4
4 9
9 6
1 6
0 5
5 7
5 8
7 2
8 3
7 9
6 8

#include <bits/stdc++.h>
using namespace std;

void fill_color(vector<int> adj[], int u, vector<int> &colors, vector<bool> &vis, vector<set<int>> &color_done){
    for(auto v : adj[u]){
        if(vis[v]) continue;
        int c = 0;
        while(color_done[v].find(c) != color_done[v].end()) c++;
        colors[v] = c;
        for(auto u_ : adj[v]){
            color_done[u_].insert(c);
        }
        vis[v] = true;
        fill_color(adj, v, colors, vis, color_done);
    }
}

int main()
{
    int V; cin>>V;
    int E; cin>>E;
    vector<int> adj[V];
    int u, v;
    for(int i=0; i<E; i++){
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> colors(V,INT_MAX);
    vector<set<int>> color_done;
    color_done.resize(V);
    vector<bool> vis(V, 0);
    vis[0] = true;
    colors[0] = 0;
    for(auto v : adj[0]){
        color_done[v].insert(0);
    }
    fill_color(adj, 0, colors, vis, color_done);
    for(auto c : colors) cout<<c<<" ";
    return 0;
}

Salesforce
2. Decode the given the string by following rules:
    a can be written as 1, b as 2,and so on till the number 9
    j can be written as 10#, k as 11# and so on till 26#.
    If the number follows this pattern: 1(2) it means ‘a’ occurs 2 times.
You have the return the array of count of each character
Ex- 110#3(3)26#
    Decoded string: ajcccz
    So the output array: 1 0 3 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
Ex- 126#(2)25#
    Decoded string: azzx
    Array: 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2

https://leetcode.com/articles/sum-of-distances-in-tree/#
https://www.hackerrank.com/contests/test-contest-27/challenges/paintwall/problem
https://www.geeksforgeeks.org/painting-fence-algorithm/
https://www.geeksforgeeks.org/painters-partition-problem/
https://www.geeksforgeeks.org/m-coloring-problem-backtracking-5/
