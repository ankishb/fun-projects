////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/longest-palindromic-subsequence/
/*
1. if s[i] == s[j] : 2 + dp[i+1][j-1] (check if we exclude these two chacter, what would be longest subsequence)
2. Else max(dp[i-1][j], dp[i][j-1])
*/
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.length(); if(n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        int j;
        for(int k=0; k<n; k++){
            for(int i=0; i<n-k; i++){
                j = i+k;
                if(i == j){ dp[i][j] = 1; continue; }
                dp[i][j] = (s[i] == s[j]) ? dp[i+1][j-1] + 2 : max(dp[i+1][j], dp[i][j-1]);
            }
        }
        // for(auto v : dp){ for(auto el : v) cout<<el<<" "; cout<<endl; }
        return dp[0][n-1];
    }
};


////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/longest-common-subsequence/
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.length(); if(n == 0) return 0;
        int m = text2.length(); if(m == 0) return 0;
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        
        for(int i=1; i<=n; i++){
            for(int j=1; j<=m; j++){
                dp[i][j] = (text1[i-1] == text2[j-1]) 
                                        ? dp[i-1][j-1] + 1 
                                        : max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[n][m];
    }
};


////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/maximum-subarray/
class Solution {
public:
    int maxSubArray(vector<int>& A) {
        int n = A.size(); if(n == 0) return 0;
        int max_sum = INT_MIN, cur_sum = 0;
        for(int i=0; i<n; i++){
            cur_sum += A[i];
            max_sum = max(max_sum, cur_sum);
            if(cur_sum < 0) cur_sum = 0;
        }
        return max_sum;
    }
};


////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/longest-turbulent-subarray/
class Solution {
public:
    int maxTurbulenceSize(vector<int>& A) {
        // return brute_force(A);
        return optimal1(A);
    }
    int brute_force(vector<int>& A) {
        int n = A.size(); if(n == 0) return 0;
        int max_turb = 1, count = 0;
        bool max_to_min;

        for(int i=0; i<n-1; i++){
            count = 1;
            max_to_min = (A[i] > A[i+1]) ? true : false;
            for(int j=i; j<n-1; j++){
                if(max_to_min && A[j] > A[j+1]){count++; max_to_min = false; }
                else if(!max_to_min && A[j] < A[j+1]){count++; max_to_min = true; }
                else break;
            }
            max_turb = max(max_turb, count);
        }
        return max_turb;
    }
    int optimal1(vector<int>& A) {
        int n = A.size(); if(n == 0) return 0;
        if(n == 1) return 1; //don't need this condition too
        int i=0, j=0;
        int max_turb = 1, count = 0;
        bool max_to_min;
        while(i<n-1){ 
            count = 1;
            max_to_min = (A[i] > A[i+1]) ? true : false;
            for(j=i; j<n-1; j++){
                if(max_to_min && A[j] > A[j+1]){count++; max_to_min = false; }
                else if(!max_to_min && A[j] < A[j+1]){count++; max_to_min = true; }
                else break;
            }
            max_turb = max(max_turb, count);
            if(i == j) i++;
            else i = j;
        }
        return max_turb;
    }
};



////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/delete-operation-for-two-strings/
class Solution {
public:
    int minDistance(string word1, string word2) {
        // return optimal_time(word1, word2);
        return optimal_space1(word1, word2);
        // return optimal_space2(word1, word2);
        
    }
    int optimal_time(string word1, string word2) {
        int n = word1.length(), m = word2.length();
        if(n == 0) return m;
        if(m == 0) return n;
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        for(int i=1; i<=n; i++){
            for(int j=1; j<=m; j++){
                dp[i][j] = (word1[i-1] == word2[j-1]) 
                           ? (1 + dp[i-1][j-1]) 
                           : max(dp[i-1][j], dp[i][j-1]);
            }
        }
        // for(auto v : dp){ for(auto el : v) cout<<el<<" "; cout<<endl; }
        // cout<<endl;
        return (n - dp[n][m]) + (m - dp[n][m]);
    }
    int optimal_space1(string s1, string s2) {
        int n = s1.length(), m = s2.length();
        if(n == 0) return m;
        if(m == 0) return n;
        vector<int> pre(m+1,0), cur(m+1, 0);
        for(int i=1; i<=n; i++){
            cur.clear(); cur.resize(m+1, 0);
            for(int j=1; j<=m; j++){
                cur[j] = (s1[i-1] == s2[j-1]) ? (1 + pre[j-1]) : max(pre[j], cur[j-1]);
            }
            pre = cur;
        }
        // for(auto v : dp){ for(auto el : v) cout<<el<<" "; cout<<endl; }
        // cout<<endl;
        return (n - cur[m]) + (m - cur[m]);
    }
    int optimal_space2(string s1, string s2) {
        int n = s1.length(), m = s2.length();
        if(n == 0) return m;
        if(m == 0) return n;
        int prev_upper, cur_upper;
        vector<int> cur(m+1, 0);
        for(int i=1; i<=n; i++){
            prev_upper = cur[0];
            for(int j=1; j<=m; j++){
                cur_upper = cur[j];
                cur[j] = (s1[i-1] == s2[j-1]) ? (1 + prev_upper) : max(cur[j], cur[j-1]);
                cur_upper = prev_upper;
            }
        }
        return (n - cur[m]) + (m - cur[m]);
    }
    
};



////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/cheapest-flights-within-k-stops/
#define iPair pair<int, int>
class Solution {
public:
    priority_queue<iPair, vector<iPair>, greater<iPair>> pq;
    vector<int> dist;
    vector<int> parent;
    vector<bool> vis;
    int u, v, w, stopsCount;
    // int minCost = INT_MAX;
    
    void minSpanningTree(vector<iPair> adj[], int src, int dst){
        dist[src] = 0;
        pq.push(make_pair(0, src));
        while(!pq.empty()){
            iPair temp = pq.top();
            pq.pop();
            u = temp.second;
            if(vis[u] == true) continue;
            vis[u] = true;
            
            for(int i=0; i<adj[u].size(); i++){
                v = adj[u][i].first, w = adj[u][i].second;
                if(vis[v] == false && dist[v] > w){
                    dist[v] = w;
                    parent[v] = u;
                    pq.push(make_pair(w, v));
                }
            }
        }
    }
    
    void getMinCostPath(vector<iPair> adj[], int src, int dst, int wts, int stops, int &minCost){
        // cout<< src <<" "<< dst <<" "<< wts <<" "<< stops <<endl;
        if(stops < -1) return ;
        if(src == dst && stops >= -1){
            minCost = min(minCost, wts);
            return ;
        }
        
        int u = src;
        for(int i=0; i<adj[u].size(); i++){
            int v = adj[u][i].first;
            // cout<< u <<" "<< v <<" "<< wts+adj[u][i].second <<" "<< stops-1 <<endl;
            // wts = wts + adj[u][i].second;
            // stops -= 1;
            getMinCostPath(adj, v, dst, wts + adj[u][i].second, stops-1, minCost);
            // wts = wts - adj[u][i].second;
            // stops += 1;
        }
    }
    
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        // return previous_solution(n, flights, src, dst, K);
        // return new_solution(n, flights, src, dst, K);
        return bellman_ford(n, flights, src, dst, K);
    }
    
    int bellman_ford(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        const int inf = INT_MAX;
        vector<int> dist(n, inf);
        dist[src] = 0;
        
        for(int i=0; i<=k; i++){
            vector<int> Dist(dist);
            for(auto e : flights){
                u = e[0], v = e[1], w = e[2];
                Dist[v] = min(Dist[v], dist[u] + w);
            }
            dist = Dist;
        }
        return dist[dst] == inf ? -1 : dist[dst];
    }
    
    
    int previous_solution(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        vector<iPair> adj[n];
        dist.resize(n, INT_MAX);
        vis.resize(n, false);
        parent.resize(n, -1);
        
        
        for(auto flight : flights){
            u=flight[0], v=flight[1], w=flight[2];
            adj[u].push_back(make_pair(v,w));
        }
        
        // minSpanningTree(adj, src, dst);
        // cout<<"dist array: ";
        // for(auto d : dist) cout<< d <<" ";
        // cout<<"\nparent array: ";
        // for(auto p : parent) cout<< p <<" ";
        // cout<<endl;
        
        // v = dst;
        // u = parent[v];
        // w = 0;
        // stopsCount = 0;
        // while(u != src){
        //     stopsCount++;
        //     w += dist[v];
        //     u = parent[u];
        // }
        // if(stopsCount <= k) return w;
        w = 0;
        int minCost = INT_MAX;
        getMinCostPath(adj, src, dst, w, K, minCost);
        return (minCost == INT_MAX) ? -1 : minCost;
    }
    
    
    
    void helper(vector<iPair> adj[], int u, int dst, int K, int steps, int cost, int &minCost, vector<bool> &vis){
        if(steps > K) return;
        if(steps <= K && u == dst){
            cout<<u<<" "<<steps<<endl;
            minCost = min(minCost, cost);
            return;
        }
        for(auto pair : adj[u]){
            if(steps < K){
                int v = pair.first;
                int w = pair.second;
                if(vis[v] == true) continue;
                vis[v] = true;
                helper(adj, v, dst, K, steps+1, cost+w, minCost, vis);
            }
        }
    }
    
    int new_solution(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        vector<iPair> adj[n];
        vis.resize(n, false);
        int minCost = INT_MAX;
        int steps = 0;
        for(auto flight : flights){
            u=flight[0], v=flight[1], w=flight[2];
            adj[u].push_back(make_pair(v,w));
        }
        helper(adj, src, dst, K+1, steps, 0, minCost, vis);
        
        return (minCost == INT_MAX) ? -1 : minCost;
    }
};



////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/maximum-subarray/discuss/20200/Share-my-solutions-both-greedy-and-divide-and-conquer
// https://leetcode.com/problems/cheapest-flights-within-k-stops/discuss/128217/Three-C%2B%2B-solutions-BFS-DFS-and-BF
// https://leetcode.com/problems/min-cost-climbing-stairs/
// https://leetcode.com/problems/divisor-game/
// https://leetcode.com/problems/first-missing-positive/




////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////




////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////





////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////




////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
void helper(vector<int> A, vector<bool> &mask, 
    vector<int> &temp, vector<vector<int>> &store){
    if(temp.size() == A.size()){
        store.push_back(temp);
        return;
    }
    for(int i=0; i<A.size(); i++){
        if(mask[i] == false){
            mask[i] = true;
            temp.push_back(A[i]);
            helper(A, mask, temp, store);
            mask[i] = false;
            temp.pop_back();
        }
    }
}

vector<vector<int> > Solution::permute(vector<int> &A) {
    int n = A.size();
    if(n == 0) return {{}};
    vector<bool> mask(n, false);
    vector<int> temp;
    vector<vector<int>> store;
    helper(A, mask, temp, store);
    return store;
}



void helper(vector<int> A, int idx, vector<int> &cur, 
    vector<vector<int>> &store){
    int n = A.size();
    if(idx == n-1){
        store.push_back(cur);
        return;
    }
    for(int i=idx; i<n; i++){
        swap(cur[idx], cur[i]);
        helper(A, idx+1, cur, store);
        swap(cur[idx], cur[i]);
    }
}

vector<vector<int> > Solution::permute(vector<int> &A) {
    int n = A.size();
    if(n <= 0) return {{}};
    if(n == 1) return {{A[0]}};
    vector<vector<int>> store;
    vector<int> cur(A);
    helper(A, 0, cur, store);
    
    return store;
}



////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
/*
* Definition for binary tree
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
void flatten_(TreeNode* root){
    if(root == NULL) return;
    if(root->left==NULL && root->right==NULL) return;
    if(root->left) flatten_(root->left);
    if(root->right) flatten_(root->right);
    
    if(root->left){
        // cout<<root->val<<" ";
        TreeNode* saveRight = root->right;
        root->right = root->left;
        root->left = NULL;
        TreeNode* saveLeft = root->right;
        while(saveLeft->right) saveLeft = saveLeft->right;
        saveLeft->right = saveRight;
    }
}

TreeNode* Recursive(TreeNode* root) {
    flatten_(root);
    return root;
}
    
TreeNode* Iterative(TreeNode* root){
    if(root == NULL) return root;
    TreeNode *save = root, *leftTree, *rightTree;
    while(root != NULL){
        if(root->left){
        // if(root->left && root->right){
            leftTree = root->left;
            rightTree = root->right;
            root->right = leftTree;
            while(leftTree->right){
                leftTree = leftTree->right;
            }
            leftTree->right = rightTree;
            root->left = NULL;
        }
        root = root->right;
    }
    return save;
}

TreeNode* Solution::flatten(TreeNode* root) {
    return Iterative(root);
    // return Recursive(root);
}
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/single-number-ii/discuss/43294/Challenge-me-thx
// https://leetcode.com/problems/first-missing-positive/






////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://www.interviewbit.com/problems/gray-code/












////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://www.interviewbit.com/problems/subset/
void helper(vector<int> A, int idx, vector<int> &cur, 
    vector<vector<int>> &store){
    int n = A.size();
    if(idx > n) return;
    for(int i=idx; i<n; i++){
        cur.push_back(A[i]);
        store.push_back(cur);
    }
    cur.clear();
    helper(A, idx+1, cur, store);
}

vector<vector<int> > Solution::subsets(vector<int> &A) {
    int n = A.size();
    if(n == 0) return {{}};
    sort(A.begin(), A.end());
    vector<int> cur;
    vector<vector<int>> store;
    store.push_back(cur);
    // for(int i=0; i<n; i++){
    helper(A, 0, cur, store);
    // }
    return store;
}




////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/find-the-difference/submissions/
class Solution {
public:
    char findTheDifference(string s, string t) {
        int n = s.length(), bit = 0;
        for(int i=0; i<n; i++){
            bit = bit ^ (int)(s[i]-'a') ^ (int)(t[i]-'a');
        }
        bit = bit ^ (t[n]-'a');
        return (char)(bit+'a');
    }
};






////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/single-number/submissions/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int n = nums.size(), bit = 0;
        for(auto a : nums) bit = bit ^ a;
        return bit;
    }
};






////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// https://leetcode.com/problems/missing-number/submissions/


class Solution {
public:
    // using sum of arary
    int constMemory1(vector<int>& nums) {
        int n = nums.size(), sum = 0;
        for(int i=0; i<n; i++) sum += nums[i];
        return (n)*(n+1)/2 - sum;
    }
    // using bit manipulation
    int constMemory2(vector<int>& nums) {
        int n = nums.size(), bit = 0;
        for(int i=0; i<n; i++){
            bit = bit ^ i ^ nums[i] ;
        }
        return bit^n;
    }
    int withSorting(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int i=0;
        while(i < n){
            if(i != nums[i]) break;
            i++;
        }
        return i;
    }
    int missingNumber(vector<int>& nums) {
        // return withSorting(nums);
        // return constMemory1(nums);
        return constMemory2(nums);
    }
};






////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
void helper(vector<int> A, int idx, vector<int> &temp, 
    set<vector<int>> &store){
    // store.insert(temp);
    for(int i=idx; i<A.size(); i++){
        temp.push_back(A[i]);
        store.insert(temp);
        helper(A, i+1, temp, store);
        temp.pop_back();
        helper(A, i+1, temp, store);
    }    
}

vector<vector<int> > recusive(vector<int> &A) {
    int n = A.size();
    if(n == 0) return {{}};
    sort(A.begin(), A.end());
    set<vector<int>> store;
    vector<vector<int>> store_;
    vector<int> temp;
    store_.push_back(temp);
    helper(A, 0, temp, store);
    
    for(auto v : store){
        store_.push_back(v);
        // for(auto el : v){
        //     cout<<el<<" ";
        // }
        // cout<<endl;
    }
    
    return store_;
}

vector<vector<int> > iterative(vector<int> &A) {
    int n = A.size();
    if(n == 0) return {{}};
    sort(A.begin(), A.end());
    vector<vector<int>> store;
    vector<int> temp;
    store.push_back(temp);
    int m = pow(2, n);

    for(int i=0; i<n; i++){
        temp.clear();
        // int k = i, j = 0, bit;
        // while(k != 0){
        //     bit = (k&1);
        //     if(bit == 1) temp.push_back(A[j]);
        //     j++;
        //     k = k>>1;
        // }
            
        for(int j=i; j<n; j++){
            temp.push_back(A[j]);
            store.push_back(temp);
        }
        
    }
    
    
    
    for(auto v : store){ for(auto el : v) cout<<el<<" "; cout<<endl; }
    
    return store;
}

vector<vector<int> > Solution::subsets(vector<int> &A) {
    return recusive(A);
    // return iterative(A);
}
