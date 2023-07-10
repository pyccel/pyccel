
## Verwendung von Header-Dateien

Eine Header-Datei in Pyccel ist eine Datei mit einem Namen, der auf `.pyh` endet, die Funktions-/Variablendeklarationen, Makrodefinitionen, Vorlagen und Metavariablendeklarationen enthält.
Header-Dateien dienen zwei Zwecken:
- Verknüpfung externer Bibliotheken in den Zielsprachen durch Bereitstellung ihrer Funktionsdeklarationen;
- Beschleunigung des Parsing-Prozesses eines importierten Python-Moduls durch Parsing nur der (automatisch generierten) Header-Datei anstelle des gesamten Moduls.

### Beispiele
#### Verknüpfung mit OpenMP
Wir erstellen die Datei `header.pyh`, die eine OpenMP-Funktionsdefinition enthält:

```python
#$ header metavar module_name = 'omp_lib'
#$ header metavar import_all = True

#$ header function omp_get_num_threads() results(int)
```
Wir erstellen dann die Datei `example.py`:

```python
from header import omp_get_num_threads
print('Anzahl der Threads ist :', omp_get_num_threads())
```
Pyccel kann die Python-Datei mit dem folgenden Befehl kompilieren: `pyccel example.py --openmp`
Es wird dann die ausführbare Datei `example` erstellt.
#### Link mit einer statischen Bibliothek
Wir haben das folgende Fortran-Modul, das wir in der Datei `funcs.f90` abgelegt haben  

``Fortran
Modul funcs

ISO_C_BINDING verwenden

implizit keine

enthält

!........................................
rekursive Funktion fib(n) Ergebnis(Ergebnis)

implizit keine

ganzzahl(C_LONG_LONG) :: ergebnis
ganze Zahl(C_LONG_LONG), Wert :: n

wenn (n < 2_C_LONG_LONG) dann
  Ergebnis = n
  return
end if
Ergebnis = fib(n - 1_C_LONG_LONG) + fib(n - 2_C_LONG_LONG)
retu
