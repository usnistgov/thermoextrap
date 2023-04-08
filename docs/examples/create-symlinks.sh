exts=(ipynb md)


rm -rf notebooks/*
for path in $(cat *.md | grep '^notebooks/'); do

    target="../../"${path}
    name=$(basename $target)

    if [ -f $target ]; then
        # has extension

        base=${name%.*}
        ext=${name##*.}

    else
        # no extension.  Try to add one
        for ext in ${exts[@]}; do
            tmp=${target}.${ext}
            if [ -f "${tmp}" ] ; then
                base=$name
                target=$tmp
                break
            fi
        done
    fi


    new_dir=$(dirname $path)
    total_target=$(realpath --relative-to=${new_dir} $target)

    echo "target $target"
    echo "base  $base"
    echo "ext  $ext"
    echo "total_target $total_target"
    echo "new_dir $new_dir"

    mkdir -p $new_dir

    ln -s $total_target $new_dir

done
