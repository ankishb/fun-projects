
/*
- [medium][saerch][connected_cell_in_grid](https://www.hackerrank.com/challenges/connected-cell-in-a-grid/problem)
- [medium][recursion][recursive_digit_sum](https://www.hackerrank.com/challenges/recursive-digit-sum/problem)
- [ ][medium][bit-manipulation][minimum](https://www.hackerrank.com/challenges/aorb/problem)
- [easy][bit-man][flipping_bits](https://www.hackerrank.com/challenges/flipping-bits/problem)
- [medium][bit][count_no_with_greater_xor](https://www.hackerrank.com/challenges/the-great-xor/problem)
- [easy][bit-manipulation][and_product_of_range](https://www.hackerrank.com/challenges/and-product/problem)
= [hard][bit-manipulation][what_next](https://www.hackerrank.com/challenges/whats-next/problem)
*/





// ===================================================
// ================flipping_bits======================
// ===================================================

long flippingBits(long n) {
    long ans = 0;
    int bit;
    for(int i=0; i<32; i++){
        bit = (n&1);
        cout<<bit<<" ";
        if(bit == 0) ans += pow(2, i);
        n = n>>1;
    }
    cout<<endl;
    return ans;
}

// ===================================================
// ===================================================
// ===================================================


// Complete the theGreatXor function below.
long theGreatXor(long x) {
    long ans = 0;
    int bit, i = 0;
    while(x > 0){
        bit = (x&1);
        if(bit == 0) ans += pow(2, i);
        i++;
        x = x>>1;
    }
    return ans;
}
// ===================================================
// ===================================================
// ===================================================
long andProduct(long a, long b) {
    long ans = a;
    a++;
    while(a != b+1){
        ans = (ans & a);
        a++;
    }
    return ans;
}

// not working
long andProduct(long a, long b) {
    long ans = 0;
    int bit_a, bit_b;
    vector<int> A, B;
    while(a > 0 || b > 0){
        bit_a = a&1;
        bit_b = b&1;
        A.push_back(bit_a);
        B.push_back(bit_b);
        a = a>>1;
        b = b>>1;
    }
    int n = A.size();
    for(int i=n-1; i>=0; i--){
        // cout<<A[i]<<" "<<B[i]<<endl;
        if(A[i] == 0 || B[i] == 0) break;
        else if(A[i] == B[i] && A[i] == 1){
            ans += pow(2, i);
        }
    }
    // for(auto itr : A) cout<<itr<<" ";
    // cout<<endl;
    // for(auto itr : B) cout<<itr<<" ";
    // cout<<endl;

    return ans;
}






// ===================================================
// ===================================================
// ===================================================


void whatsNext(vector<int> arr) {
    int n = arr.size();
    int odd, even, save_odd;
    // cout<<n<<" "<<arr.back()<<" ";
    // last is even number
    if(n%2 != 0){
        even = arr.back();
        arr.pop_back();
        odd = arr.back();
        arr.pop_back();
    }
    // last is odd number, so we get even number from the n-1
    else{
        // cout<<n<<" "<<arr.back()<<" ";
        save_odd = arr.back();
        arr.pop_back();
        even = arr.back();
        arr.pop_back();
        odd = arr.back();
        arr.pop_back();
    }

    // make logic
    if(odd>1 && even>1){
        // cout<<"case 1: ";
        arr.push_back(odd-1);
        arr.push_back(1);
        arr.push_back(1);
        arr.push_back(even-1);
        if(n%2 == 0) arr.push_back(save_odd);
    }
    else if(odd>1 && even==1){
        // cout<<"case 2: ";
        arr.push_back(odd-1);
        arr.push_back(1);
        // cout<<save_odd<<" ";
        if(n%2 == 0) arr.push_back(save_odd+1);
        else arr.push_back(1);
    }
    else if(odd == 1 && even > 1){
        // cout<<"case 3: ";
        int temp = arr.back();
        arr.pop_back();
        arr.push_back(temp+1);
        arr.push_back(1);
        arr.push_back(even-1);
        if(n%2 == 0) arr.push_back(save_odd);
    }
    else{
        // cout<<"case 4: ";
        int temp = arr.back();
        arr.pop_back();
        arr.push_back(temp+1);
        if(n%2 == 0) arr.push_back(save_odd+1);
        else arr.push_back(1);
    }
    cout<<arr.size()<<endl;
    for(auto itr : arr) cout<<itr<<" ";
    cout<<endl;
}





// ===================================================
// ===================================================
// ===================================================









// ===================================================
// ===================================================
// ===================================================











// ===================================================
// ===================================================
// ===================================================









// ===================================================
// ===================================================
// ===================================================







// ===================================================
// ===================================================
// ===================================================
// ===================================================
// ====================connectedCell==================
// ===================================================

int getCount(vector<vector<int>> &A, int i, int j, int &count){
    int n = A.size(), m = A[0].size();
    int xs[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    int ys[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    int x, y;
    count = count+1;
    A[i][j] = 0;

    for(int k=0; k<8; k++){
        x = i + xs[k], y = j + ys[k];
        if(x>=0 && x<n && y>=0 && y<m && A[x][y]==1){
            // cout<<x<<", "<<y<<endl;
            getCount(A, x, y, count);
        }
    }
    return count;
}

// Complete the connectedCell function below.
int connectedCell(vector<vector<int>> matrix) {
    int n = matrix.size();
    if(n == 0) return 0;
    int m = matrix[0].size();
    if(m == 0) return 0;

    int maxCell = 0, curCell = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(matrix[i][j] == 1){
                curCell = 0;
                getCount(matrix, i, j, curCell);
                maxCell = max(maxCell, curCell);
            }
        }
    }
    return maxCell;
}

// ===================================================
// ==================superDigit=======================
// ===================================================



int reduce(int n){
    // cout<<n<<endl;
    if(n%10 == n) return n;
    int ans = 0;
    while(n > 0){
        ans += (n%10);
        n = n/10;
    }
    return reduce(ans);
}

// Complete the superDigit function below.
int superDigit(string n, int k) {
    // cout<<n<<" "<<k<<endl;
    if(n.length() == 0) return reduce(k);
    if(n.length() == 1) return superDigit("", (n[0]-'0')*k);
    long long int ans = 0;
    for(int i=0; i<n.length(); i++){
        ans += (n[i] - '0');
    }
    n = "";
    while(ans > 0){
        n = to_string(ans%10) + n; 
        ans = ans/10;
    }
    return superDigit(n, k);
}


// ===================================================
// ===================================================
// ===================================================




string charToHex(char x){
    unordered_map<char, string> myMap;
    myMap['0'] = "0000"; myMap['6'] = "0110"; myMap['C'] = "1100";
    myMap['1'] = "0001"; myMap['7'] = "0111"; myMap['D'] = "1101";
    myMap['2'] = "0010"; myMap['8'] = "1000"; myMap['E'] = "1110";
    myMap['3'] = "0011"; myMap['9'] = "1001"; myMap['F'] = "1111";
    myMap['4'] = "0100"; myMap['A'] = "1010";
    myMap['5'] = "0101"; myMap['B'] = "1011";
    
    return myMap[x];
}

string hexToChar(string x){
    unordered_map<string, string> myMap;
    myMap["0000"] = "0"; myMap["0110"] = "6"; myMap["1100"] = "C";
    myMap["0001"] = "1"; myMap["0111"] = "7"; myMap["1101"] = "D";
    myMap["0010"] = "2"; myMap["1000"] = "8"; myMap["1110"] = "E";
    myMap["0011"] = "3"; myMap["1001"] = "9"; myMap["1111"] = "F";
    myMap["0100"] = "4"; myMap["1010"] = "A";
    myMap["0101"] = "5"; myMap["1011"] = "B";

    return myMap[x];
}

void fillMatrix(vector<vector<int>> &matrix, string a, string b, string c){
    string res;

    res = charToHex(a[0]);
    for(int i=0; i<4; i++) matrix[0][i] = (res[i]-'0');
    res = charToHex(a[1]);
    for(int i=0; i<4; i++) matrix[0][i+4] = (res[i]-'0');

    res = charToHex(b[0]);
    for(int i=0; i<4; i++) matrix[1][i] = (res[i]-'0');
    res = charToHex(b[1]);
    for(int i=0; i<4; i++) matrix[1][i+4] = (res[i]-'0');

    res = charToHex(c[0]);
    for(int i=0; i<4; i++) matrix[2][i] = (res[i]-'0');
    res = charToHex(c[1]);
    for(int i=0; i<4; i++) matrix[2][i+4] = (res[i]-'0');

}

void aOrB(int k, string a, string b, string c) {
    vector<vector<int>> A(8, vector<int>(3,0));
    fillMatrix(A, a, b, c);
    for(auto itr : A){
        cout<<itr[0]<<" "<<itr[1]<<" "<<itr[2]<<endl;
    }
    // int count = 0;
    // for(int i=0; i<8; i++){
    //     if(A[2][i] == 0){
    //         if(A[0][i] == 1){ A[0][i]=0; count++; }
    //         if(A[1][i] == 1){ A[1][i]=0; count++; }
    //     }
    //     else{
    //         if(A[0][i]==0 && A[1][i]==0){ A[1][i]=1; count++; }
    //         if(A[0][i]==0 && A[1][i]==1){ continue; }
    //         if(A[0][i]==1 && A[1][i]==0){ continue; }
    //         if(A[0][i]==1 && A[1][i]==1){ continue; }
    //     }
    // }
    // if(count > k) cout<<-1<<endl;
    // else{
    //     string a1="", b1="";
    //     int a11, b11;
    //     a11 = 0, b11 = 0;
    //     for(int i=0; i<4; i++){
    //         a11 = a11*10 + A[0][i];
    //         b11 = b11*10 + A[1][i];
            
    //     }
    //     a1 += hexToChar(to_string(a11));
    //     b1 += hexToChar(to_string(b11));
    //     a11 = 0, b11 = 0;
    //     for(int i=4; i<8; i++){
    //         a11 = a11*10 + A[0][i];
    //         b11 = b11*10 + A[1][i];
    //     }
    //     a1 += hexToChar(to_string(a11));
    //     b1 += hexToChar(to_string(b11));

    //     cout<<a1<<endl<<b1<<endl;
    // }
    
}

// ===================================================
// ===================================================
// ===================================================


