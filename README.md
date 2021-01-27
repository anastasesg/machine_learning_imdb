# Machine Learning - IMDB
Το πρόγραμμα αποτελείται από τρία αρχεία: main.py, processing.py και logistic_regression.py.

# Προεπεξερασία - preprocessing.py
Στο κομμάτι αυτό του προγράμματος υλοποιείται η προεπεξεργασία του κειμένου.
- Μέθοδος `__init__`: Αρχικοποιεί την κλάσση `Data` με τις κενές μεταβλητές `self.examples`, `self.labels`, `self.vocabulary`, `self.attributes` και `self.data_frame`.
- Μέθοδος `read_vocab`: Διαβάζει το αρχείο `imdb.vocab`, εξαγάγει τις λέξεις που περιέχει και τις αποθηκεύει σε ένα `numpy.array`. Παίρνει ως όρισμα τη διαδρομή ενός αρχείου που περιέχει το λεξιλόγιο `filename`.
- Στατική μέθοδος `example_open`: Συμπυκνώνει το άνοιγμα, διάβασμα, κλείσιμο και την τυποποίηση του κάθε παραδείγματος εκπαίδευσης. Παίρνει ως όρισμα τη διαδρομή κάποιου παραδείγματος εκπαίδευσης `filename`.
- Μέθοδος `read_examples`: Διαβάζει κάθε παράδειγμα από τους φακέλους `train/pos`, `train/neg`, `test/pos` και `test/neg` με την βοήθεια της μεθόδου `example_open`. Στην συνέχεια τα αποθηκεύει σε ένα αρχείο `.csv` για ευκολία ανάγνωσης σε μελλοντικές χρήσεις του προγράμματος. Παίρνει ως όρισμα τον υπερφάκελο που περιέχει τα παραδείγματα εκπαίδευσης `parent_directory`.
- Στατική μέθοδος `information_gain`: Υπολογίζει το κέρδος πληροφορίας μίας δοσμένης λέξης. Παίρνει ως όρισμα μία λέξη `word` και επιστρέφει μία τιμή `0-1`.
- Μέθοδος `select_attributes`: Υπολογίζει το κέρδος πληροφορίας της κάθε λέξης του λεξιλογίου και τα αποθηκεύει σε ένα αρχείο `.csv` για ευκολία ανάγνωσης σε μελλοντικά χρήσεις του προγράμματος. Παίρνει ως όρισμα το επιθυμητό πλήθος ιδιοτήτων `max_length`.
- Μέθοδος `vectorize`: Μετατρέπει κάθε παράδειγμα εκπαίδευσης σε ένα διάνυσμα `max_length x 1` με το `n`-οστό στοιχείο του να είναι 1 αν η `n`-οστή λέξη υπάρχει και 0 διαφορετικά.
- Μέθοδος `load_data`: Συνδυάζονται οι προηγούμενες μέθοδοι για την ανάγνωση του λεξιλογίου και των παραδειγμάτων, η επιλογή των ιδιοτήτων και η διανυσματικοποίηση. Επιστρέφει τα παραδείγματα και τις αντίστοιχες κατηγορίες τους.
- Στατική μέθοδος `data_split`: Χωρίζει τα παραδείγματα εκπαίδευσης και τις αντίστοιχες κατηγορίες τους σε δεδομένα εκπαίδευσης και ελέγχου. Παίρνει ως όρισμα το ποσοστό του συνολικού πλήθους των παραδειγμάτων που θα χρησιμοποιηθούν ως παραδείγματα ελέγχου `split`.

# Λογιστική παλινδρόμηση - logistic_regression.py

# Εκτέλεση και αξιολόγηση - main.py

# Χρήση και προαπαιτούμενα
