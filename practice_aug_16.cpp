
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
