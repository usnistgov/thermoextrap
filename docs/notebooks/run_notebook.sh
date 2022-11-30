for x in $@; do
    echo "Working on $x"
    jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --execute $x
    echo "done"
done
