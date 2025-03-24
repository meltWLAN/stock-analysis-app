// 全局变量
let isLoading = false;

// 文档加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化搜索功能
    initSearchFunctionality();
    
    // 初始化移动端搜索模态框搜索功能
    initModalSearch();
    
    // 加载遮罩操作
    setupLoadingOverlay();
});

// 初始化搜索功能
function initSearchFunctionality() {
    const searchInput = document.getElementById('search-input');
    const searchForm = document.getElementById('search-form');
    const suggestionsContainer = document.getElementById('search-suggestions');
    
    if (!searchInput || !searchForm || !suggestionsContainer) return;
    
    // 监听输入变化
    searchInput.addEventListener('input', debounce(function() {
        const keyword = this.value.trim();
        
        if (keyword.length < 2) {
            suggestionsContainer.classList.add('d-none');
            return;
        }
        
        fetchSearchSuggestions(keyword, suggestionsContainer);
    }, 300));
    
    // 提交搜索表单
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const code = searchInput.value.trim();
        if (code) {
            navigateToStockDetail(code);
        }
    });
    
    // 点击其他区域关闭搜索建议
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
            suggestionsContainer.classList.add('d-none');
        }
    });
}

// 初始化模态框搜索
function initModalSearch() {
    const modalSearchInput = document.getElementById('modal-search-input');
    const modalSearchForm = document.getElementById('modal-search-form');
    const modalSuggestionsContainer = document.getElementById('modal-search-suggestions');
    
    if (!modalSearchInput || !modalSearchForm || !modalSuggestionsContainer) return;
    
    // 监听输入变化
    modalSearchInput.addEventListener('input', debounce(function() {
        const keyword = this.value.trim();
        
        if (keyword.length < 2) {
            modalSuggestionsContainer.classList.add('d-none');
            return;
        }
        
        fetchSearchSuggestions(keyword, modalSuggestionsContainer);
    }, 300));
    
    // 提交搜索表单
    modalSearchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const code = modalSearchInput.value.trim();
        if (code) {
            navigateToStockDetail(code);
        }
    });
    
    // 点击其他区域关闭搜索建议
    document.addEventListener('click', function(e) {
        if (!modalSearchInput.contains(e.target) && !modalSuggestionsContainer.contains(e.target)) {
            modalSuggestionsContainer.classList.add('d-none');
        }
    });
}

// 获取搜索建议
function fetchSearchSuggestions(keyword, suggestionsContainer) {
    // 显示加载中
    suggestionsContainer.innerHTML = '<div class="p-2 text-center"><div class="spinner-border spinner-border-sm" role="status"></div></div>';
    suggestionsContainer.classList.remove('d-none');
    
    // 发送搜索请求
    fetch(`/api/stock/search?keyword=${encodeURIComponent(keyword)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.length > 0) {
                // 清空容器
                suggestionsContainer.innerHTML = '';
                
                // 添加建议项
                data.data.forEach(item => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.className = 'suggestion-item';
                    suggestionItem.innerHTML = `
                        <span>${item.name}</span>
                        <span class="suggestion-code">${item.code}</span>
                    `;
                    suggestionItem.addEventListener('click', function() {
                        navigateToStockDetail(item.code);
                    });
                    
                    suggestionsContainer.appendChild(suggestionItem);
                });
            } else {
                suggestionsContainer.innerHTML = '<div class="p-2 text-center text-muted">无匹配结果</div>';
            }
        })
        .catch(error => {
            console.error('搜索请求出错:', error);
            suggestionsContainer.innerHTML = '<div class="p-2 text-center text-danger">请求失败，请重试</div>';
        });
}

// 跳转到股票详情页
function navigateToStockDetail(code) {
    // 清空搜索框
    const searchInput = document.getElementById('search-input');
    const modalSearchInput = document.getElementById('modal-search-input');
    if (searchInput) searchInput.value = '';
    if (modalSearchInput) modalSearchInput.value = '';
    
    // 隐藏建议
    const suggestionsContainer = document.getElementById('search-suggestions');
    const modalSuggestionsContainer = document.getElementById('modal-search-suggestions');
    if (suggestionsContainer) suggestionsContainer.classList.add('d-none');
    if (modalSuggestionsContainer) modalSuggestionsContainer.classList.add('d-none');
    
    // 关闭模态框
    const searchModal = document.getElementById('searchModal');
    if (searchModal) {
        const bsModal = bootstrap.Modal.getInstance(searchModal);
        if (bsModal) bsModal.hide();
    }
    
    // 显示加载状态
    showLoading();
    
    // 跳转到详情页
    window.location.href = `/stock?code=${encodeURIComponent(code)}`;
}

// 加载遮罩设置
function setupLoadingOverlay() {
    // 查找加载遮罩元素
    const loadingOverlay = document.getElementById('loading-overlay');
    if (!loadingOverlay) return;
    
    // 监听页面加载完成事件
    window.addEventListener('load', function() {
        hideLoading();
    });
}

// 显示加载遮罩
function showLoading() {
    isLoading = true;
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('d-none');
    }
}

// 隐藏加载遮罩
function hideLoading() {
    isLoading = false;
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.classList.add('d-none');
    }
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

// 添加收藏
function addToFavorites(stockCode, stockName) {
    const favorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    
    // 检查是否已存在
    const existingIndex = favorites.findIndex(item => item.code === stockCode);
    if (existingIndex !== -1) {
        return false; // 已经收藏过了
    }
    
    // 添加到收藏
    favorites.push({ code: stockCode, name: stockName });
    localStorage.setItem('favorites', JSON.stringify(favorites));
    
    return true;
}

// 移除收藏
function removeFromFavorites(stockCode) {
    const favorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    
    // 找到并移除
    const newFavorites = favorites.filter(item => item.code !== stockCode);
    
    if (newFavorites.length !== favorites.length) {
        localStorage.setItem('favorites', JSON.stringify(newFavorites));
        return true;
    }
    
    return false;
}

// 检查是否已收藏
function isFavorite(stockCode) {
    const favorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    return favorites.some(item => item.code === stockCode);
}

// 切换收藏状态
function toggleFavorite(stockCode, stockName, btnElement) {
    if (isFavorite(stockCode)) {
        if (removeFromFavorites(stockCode)) {
            if (btnElement) {
                btnElement.innerHTML = '<i class="far fa-star"></i>';
                btnElement.setAttribute('title', '加入收藏');
            }
            showToast('已从收藏移除');
        }
    } else {
        if (addToFavorites(stockCode, stockName)) {
            if (btnElement) {
                btnElement.innerHTML = '<i class="fas fa-star"></i>';
                btnElement.setAttribute('title', '取消收藏');
            }
            showToast('已加入收藏');
        }
    }
}

// 显示Toast消息
function showToast(message) {
    // 创建Toast元素
    const toastContainer = document.createElement('div');
    toastContainer.style.position = 'fixed';
    toastContainer.style.bottom = '20px';
    toastContainer.style.left = '50%';
    toastContainer.style.transform = 'translateX(-50%)';
    toastContainer.style.zIndex = '9999';
    
    toastContainer.innerHTML = `
        <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-body">${message}</div>
        </div>
    `;
    
    document.body.appendChild(toastContainer);
    
    // 2秒后移除
    setTimeout(() => {
        toastContainer.remove();
    }, 2000);
} 