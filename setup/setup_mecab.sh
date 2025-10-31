#!/bin/bash
set -e

echo "==================================="
echo "MeCab Setup Script (No sudo)"
echo "==================================="

# インストール先ディレクトリ
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="${SCRIPT_DIR}/lib/mecab"
BIN_DIR="${SCRIPT_DIR}/bin"
BUILD_DIR="${SCRIPT_DIR}/.mecab_build"

# MeCabバージョン
MECAB_VERSION="0.996"
MECAB_DICT_VERSION="2.1.1-20180310"

echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo ""

# 既存のインストールを確認
if [ -f "${BIN_DIR}/mecab" ]; then
    echo "✓ MeCab is already installed"
    "${BIN_DIR}/mecab" --version
    echo ""
    
    # 既に環境変数が設定されているか確認
    if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
        echo "Adding MeCab to PATH..."
        export PATH="${BIN_DIR}:${PATH}"
        export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    fi
else
    echo "Installing MeCab from source (no sudo required)..."
    echo ""
    
    # ビルドディレクトリを作成
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    # MeCab本体のダウンロードとインストール
    if [ ! -f "mecab-${MECAB_VERSION}.tar.gz" ]; then
        echo "Downloading MeCab ${MECAB_VERSION}..."
        wget -q "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE" -O "mecab-${MECAB_VERSION}.tar.gz"
    fi
    
    echo "Extracting MeCab..."
    tar xzf "mecab-${MECAB_VERSION}.tar.gz"
    
    echo "Building MeCab..."
    cd "mecab-${MECAB_VERSION}"
    ./configure --prefix="${INSTALL_DIR}" --with-charset=utf8
    make -j$(nproc 2>/dev/null || echo 2)
    make install
    
    cd "${BUILD_DIR}"
    
    # IPA辞書のダウンロードとインストール
    if [ ! -f "mecab-ipadic-${MECAB_DICT_VERSION}.tar.gz" ]; then
        echo ""
        echo "Downloading MeCab IPA dictionary..."
        wget -q "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM" -O "mecab-ipadic-${MECAB_DICT_VERSION}.tar.gz"
    fi
    
    echo "Extracting IPA dictionary..."
    tar xzf "mecab-ipadic-${MECAB_DICT_VERSION}.tar.gz"
    
    echo "Building IPA dictionary..."
    cd "mecab-ipadic-${MECAB_DICT_VERSION}"
    ./configure --prefix="${INSTALL_DIR}" --with-mecab-config="${INSTALL_DIR}/bin/mecab-config" --with-charset=utf8
    make -j$(nproc 2>/dev/null || echo 2)
    make install
    
    # binディレクトリへのシンボリックリンクを作成
    mkdir -p "${BIN_DIR}"
    ln -sf "${INSTALL_DIR}/bin/mecab" "${BIN_DIR}/mecab"
    ln -sf "${INSTALL_DIR}/bin/mecab-config" "${BIN_DIR}/mecab-config"
    
    # クリーンアップ
    cd "${SCRIPT_DIR}"
    rm -rf "${BUILD_DIR}"
    
    echo ""
    echo "✓ MeCab installed successfully"
    
    # 環境変数を設定
    export PATH="${BIN_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    
    "${BIN_DIR}/mecab" --version
    echo ""
fi

# mecabrcの確認
echo "Checking MeCab configuration..."
MECABRC="${INSTALL_DIR}/etc/mecabrc"
if [ -f "${MECABRC}" ]; then
    echo "✓ Found mecabrc at: ${MECABRC}"
    export MECABRC="${MECABRC}"
else
    echo "⚠ mecabrc not found at expected location"
fi

# 辞書の場所を確認
echo ""
echo "Checking MeCab dictionary..."
"${BIN_DIR}/mecab" -D 2>/dev/null || echo "Note: Run 'mecab -D' to see dictionary information"

echo ""
echo "Testing MeCab..."
echo "これはテストです。" | "${BIN_DIR}/mecab" -Owakati

echo ""
echo "==================================="
echo "MeCab Setup Complete!"
echo "==================================="
echo ""

# 環境変数設定スクリプトの作成
ENV_FILE="${SCRIPT_DIR}/mecab_env.sh"
cat > "${ENV_FILE}" << EOF
# Source this file to add MeCab to your environment
# Usage: source ${ENV_FILE}

export PATH="${BIN_DIR}:\${PATH}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}"
export MECABRC="${MECABRC}"
EOF

echo "Created environment file: ${ENV_FILE}"
echo "To use MeCab in new shells, run:"
echo "  source ${ENV_FILE}"
echo ""

# Python mecab-python3のインストール
if [ -d "../.venv" ]; then
    echo "Installing mecab-python3 in virtual environment..."
    source "../.venv/bin/activate"
    
    # mecab-configのパスを設定
    export PATH="${BIN_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    
    pip install mecab-python3
    echo "✓ mecab-python3 installed"
    echo ""
elif [ -d ".venv" ]; then
    echo "Installing mecab-python3 in virtual environment..."
    source ".venv/bin/activate"
    
    export PATH="${BIN_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    
    pip install mecab-python3
    echo "✓ mecab-python3 installed"
    echo ""
fi

# テスト
echo "Testing Python MeCab binding..."
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
    export PATH="${BIN_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    export MECABRC="${MECABRC}"
    
    python3 << EOF
try:
    import MeCab
    tagger = MeCab.Tagger("-Owakati")
    result = tagger.parse("これはテストです。")
    print("✓ Python MeCab test successful")
    print(f"  Result: {result.strip()}")
except Exception as e:
    print(f"✗ Python MeCab test failed: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ All tests passed!"
    else
        echo ""
        echo "✗ Python MeCab test failed"
        echo "Please check the error message above"
        exit 1
    fi
elif [ -d ".venv" ]; then
    source ".venv/bin/activate"
    export PATH="${BIN_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    export MECABRC="${MECABRC}"
    
    python3 << EOF
try:
    import MeCab
    tagger = MeCab.Tagger("-Owakati")
    result = tagger.parse("これはテストです。")
    print("✓ Python MeCab test successful")
    print(f"  Result: {result.strip()}")
except Exception as e:
    print(f"✗ Python MeCab test failed: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ All tests passed!"
    else
        echo ""
        echo "✗ Python MeCab test failed"
        echo "Please check the error message above"
        exit 1
    fi
else
    echo "Virtual environment not found. Skipping Python test."
fi

echo ""
echo "MeCab is ready to use!"
echo ""
echo "IMPORTANT: To use MeCab in your shell, run:"
echo "  source ${ENV_FILE}"
