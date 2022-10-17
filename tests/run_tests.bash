if hash python3 2>/dev/null; then
    python3 -m unittest -b
elif hash python 2>/dev/null; then
    if python -c 'import sys; print(sys.version_info[0])' | grep -q 3; then
        python -m unittest -b
    else
        echo "Python detected, but it must be version 3.x"
    fi
else
    echo "Python not found"
fi
