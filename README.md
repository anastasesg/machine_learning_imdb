# Machine Learning - IMDB
Το πρόγραμμα αποτελείται από τρία αρχεία: main.py, processing.py και logistic_regression.py.

# Προεπεξερασία - preprocessing.py
Στο κομμάτι αυτό του προγράμματος υλοποιείται η προεπεξεργασία του κειμένου.
- Μέθοδος `__init__`: Αρχικοποιεί την κλάσση `Data` με τις κενές μεταβλητές `self.examples`, `self.labels`, `self.vocabulary`, `self.attributes` και `self.data_frame`.
- Μέθοδος `read_examples`: Διαβάζει το αρχείο `imdb.vocab`, εξαγάγει τις λέξεις που περιέχει και τις αποθηκεύει σε ένα `numpy.array`.
- Στατική μέθοδος `example_open`: Συμπυκνώνει το άνοιγμα, διάβασμα, κλείσιμο και την τυποποίηση του κάθε παραδείγματος εκπαίδευσης.
- Μέθοδος `read_examples`: Ανοίγει 

# Λογιστική παλινδρόμηση - logistic_regression.py

# Εκτέλεση και αξιολόγηση - main.py
