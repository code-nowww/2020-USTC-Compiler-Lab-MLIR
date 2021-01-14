int main(){
    int i;
    L1:
        if(i)
            goto L2;
        else
        goto L3;
    L2:
        goto L1;
    L3:
        goto L1;
}