"""tools/src/*.py 로부터 주차별 demo.ipynb 를 전부 다시 생성한다.

    python tools/build.py
"""

from nbgen import build_all

if __name__ == "__main__":
    print("노트북 생성:")
    build_all()
    print("\n완료. Colab에 올리기 전에 로컬에서 한 번 열어 셀 구분이 맞는지만 확인할 것.")
