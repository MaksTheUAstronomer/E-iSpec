objects=("J065127" "J194853")
#objects=("SZMonT+" "SZMonT-" "SZMonG+" "SZMonG-" "SZMonM+" "SZMonM-" "SZMonV+" "SZMonV-") # for error estimation
#objects=("DFCygT+" "DFCygT-" "DFCygG+" "DFCygG-" "DFCygM+" "DFCygM-" "DFCygV+" "DFCygV-") # for error estimation

for obj in ${objects[@]}; do # the order of scripts (not all are needed to run every time)
    python3 PrepSpec.py $obj
    python3 FindSpecWinds.py $obj
    python3 PlotSpecWinds.py $obj
    python3 UseSpecWinds.py $obj #python3 UseSpecWinds_uncertainties.py $obj
    python3 ComparWithLineLists.py $obj
done
