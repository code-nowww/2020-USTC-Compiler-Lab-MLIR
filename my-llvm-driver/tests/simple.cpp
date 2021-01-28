int main() {
    int i = 0;
    while (i > 5) {
        i = i + 4;
        for (int j = 0; j < 6; j++) {
            j = i + 1;
        }
    }
}