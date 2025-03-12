// 打印主题标识,请保留出处
;(function () {
  var style1 = 'background:#4BB596;color:#ffffff;border-radius: 2px;'
  var style2 = 'color:auto;'
  var author = ' TMaize'
  var github = ' https://github.com/TMaize/tmaize-blog'
  var build = ' ' + blog.buildAt.substr(0, 4)
  build += '/' + blog.buildAt.substr(4, 2)
  build += '/' + blog.buildAt.substr(6, 2)
  build += ' ' + blog.buildAt.substr(8, 2)
  build += ':' + blog.buildAt.substr(10, 2)
  console.info('%c Author %c' + author, style1, style2)
  console.info('%c Build  %c' + build, style1, style2)
  console.info('%c GitHub %c' + github, style1, style2)
})()

/**
 * 工具，允许多次onload不被覆盖
 * @param {方法} func
 */
blog.addLoadEvent = function (func) {
  var oldonload = window.onload
  if (typeof window.onload != 'function') {
    window.onload = func
  } else {
    window.onload = function () {
      oldonload()
      func()
    }
  }
}

/**
 * 工具，兼容的方式添加事件
 * @param {单个DOM节点} dom
 * @param {事件名} eventName
 * @param {事件方法} func
 * @param {是否捕获} useCapture
 */
blog.addEvent = function (dom, eventName, func, useCapture) {
  if (window.attachEvent) {
    dom.attachEvent('on' + eventName, func)
  } else if (window.addEventListener) {
    if (useCapture != undefined && useCapture === true) {
      dom.addEventListener(eventName, func, true)
    } else {
      dom.addEventListener(eventName, func, false)
    }
  }
}

/**
 * 工具，DOM添加某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.addClass = function (dom, className) {
  if (!blog.hasClass(dom, className)) {
    var c = dom.className || ''
    dom.className = c + ' ' + className
    dom.className = blog.trim(dom.className)
  }
}

/**
 * 工具，DOM是否有某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.hasClass = function (dom, className) {
  var list = (dom.className || '').split(/\s+/)
  for (var i = 0; i < list.length; i++) {
    if (list[i] == className) return true
  }
  return false
}

/**
 * 工具，DOM删除某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.removeClass = function (dom, className) {
  if (blog.hasClass(dom, className)) {
    var list = (dom.className || '').split(/\s+/)
    var newName = ''
    for (var i = 0; i < list.length; i++) {
      if (list[i] != className) newName = newName + ' ' + list[i]
    }
    dom.className = blog.trim(newName)
  }
}

/**
 * 工具，兼容问题，某些OPPO手机不支持ES5的trim方法
 * @param {字符串} str
 */
blog.trim = function (str) {
  return str.replace(/^\s+|\s+$/g, '')
}

/**
 * 工具，转义html字符
 * @param {字符串} str
 */
blog.htmlEscape = function (str) {
  var temp = document.createElement('div')
  temp.innerText = str
  str = temp.innerHTML
  temp = null
  return str
}

/**
 * 工具，转换实体字符防止XSS
 * @param {字符串} str
 */
blog.encodeHtml = function (html) {
  var o = document.createElement('div')
  o.innerText = html
  var temp = o.innerHTML
  o = null
  return temp
}

/**
 * 工具， 转义正则关键字
 * @param {字符串} str
 */
blog.encodeRegChar = function (str) {
  // \ 必须在第一位
  var arr = ['\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')']
  arr.forEach(function (c) {
    var r = new RegExp('\\' + c, 'g')
    str = str.replace(r, '\\' + c)
  })
  return str
}

/**
 * 工具，Ajax
 * @param {字符串} str
 */
blog.ajax = function (option, success, fail) {
  var xmlHttp = null
  if (window.XMLHttpRequest) {
    xmlHttp = new XMLHttpRequest()
  } else {
    xmlHttp = new ActiveXObject('Microsoft.XMLHTTP')
  }
  var url = option.url
  var method = (option.method || 'GET').toUpperCase()
  var sync = option.sync === false ? false : true
  var timeout = option.timeout || 10000

  var timer
  var isTimeout = false
  xmlHttp.open(method, url, sync)
  xmlHttp.onreadystatechange = function () {
    if (isTimeout) {
      fail({
        error: '请求超时'
      })
    } else {
      if (xmlHttp.readyState == 4) {
        if (xmlHttp.status == 200) {
          success(xmlHttp.responseText)
        } else {
          fail({
            error: '状态错误',
            code: xmlHttp.status
          })
        }
        //清除未执行的定时函数
        clearTimeout(timer)
      }
    }
  }
  timer = setTimeout(function () {
    isTimeout = true
    fail({
      error: '请求超时'
    })
    xmlHttp.abort()
  }, timeout)
  xmlHttp.send()
}

/**
 * 特效：点击页面文字冒出特效
 */
blog.initClickEffect = function (textArr) {
  function createDOM(text) {
    var dom = document.createElement('span')
    dom.innerText = text
    dom.style.left = 0
    dom.style.top = 0
    dom.style.position = 'fixed'
    dom.style.fontSize = '12px'
    dom.style.whiteSpace = 'nowrap'
    dom.style.webkitUserSelect = 'none'
    dom.style.userSelect = 'none'
    dom.style.opacity = 0
    dom.style.transform = 'translateY(0)'
    dom.style.webkitTransform = 'translateY(0)'
    return dom
  }

  blog.addEvent(window, 'click', function (ev) {
    let target = ev.target
    while (target !== document.documentElement) {
      if (target.tagName.toLocaleLowerCase() == 'a') return
      if (blog.hasClass(target, 'footer-btn')) return
      target = target.parentNode
    }

    var text = textArr[parseInt(Math.random() * textArr.length)]
    var dom = createDOM(text)

    document.body.appendChild(dom)
    var w = parseInt(window.getComputedStyle(dom, null).getPropertyValue('width'))
    var h = parseInt(window.getComputedStyle(dom, null).getPropertyValue('height'))

    var sh = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0
    dom.style.left = ev.pageX - w / 2 + 'px'
    dom.style.top = ev.pageY - sh - h + 'px'
    dom.style.opacity = 1

    setTimeout(function () {
      dom.style.transition = 'transform 500ms ease-out, opacity 500ms ease-out'
      dom.style.webkitTransition = 'transform 500ms ease-out, opacity 500ms ease-out'
      dom.style.opacity = 0
      dom.style.transform = 'translateY(-26px)'
      dom.style.webkitTransform = 'translateY(-26px)'
    }, 20)

    setTimeout(function () {
      document.body.removeChild(dom)
      dom = null
    }, 520)
  })
}

// 新建DIV包裹TABLE
blog.addLoadEvent(function () {
  // 文章页生效
  if (document.getElementsByClassName('page-post').length == 0) {
    return
  }
  var tables = document.getElementsByTagName('table')
  for (var i = 0; i < tables.length; i++) {
    var table = tables[i]
    var elem = document.createElement('div')
    elem.setAttribute('class', 'table-container')
    table.parentNode.insertBefore(elem, table)
    elem.appendChild(table)
  }
})

// 回到顶部
blog.addLoadEvent(function () {
  var el = document.querySelector('.footer-btn.to-top')
  if (!el) return
  function getScrollTop() {
    if (document.documentElement && document.documentElement.scrollTop) {
      return document.documentElement.scrollTop
    } else if (document.body) {
      return document.body.scrollTop
    }
  }
  function ckeckToShow() {
    if (getScrollTop() > 200) {
      blog.addClass(el, 'show')
    } else {
      blog.removeClass(el, 'show')
    }
  }
  blog.addEvent(window, 'scroll', ckeckToShow)
  blog.addEvent(
    el,
    'click',
    function (event) {
      window.scrollTo(0, 0)
      event.stopPropagation()
    },
    true
  )
  ckeckToShow()
})

// 点击图片全屏预览
blog.addLoadEvent(function () {
  if (!document.querySelector('.page-post')) {
    return
  }
  console.debug('init post img click event')
  let imgMoveOrigin = null
  let restoreLock = false
  let imgArr = document.querySelectorAll('.page-post img')

  let css = [
    '.img-move-bg {',
    '  transition: opacity 300ms ease;',
    '  position: fixed;',
    '  left: 0;',
    '  top: 0;',
    '  right: 0;',
    '  bottom: 0;',
    '  opacity: 0;',
    '  background-color: #000000;',
    '  z-index: 1000;',
    '}',
    '.img-move-item {',
    '  transition: all 300ms ease;',
    '  position: fixed;',
    '  opacity: 0;',
    '  cursor: pointer;',
    '  z-index: 1001;',
    '  box-shadow: 0 5px 25px rgba(0,0,0,0.3);',
    '  border: 1px solid rgba(255,255,255,0.2);',
    '  max-width: 100%;',
    '  max-height: 100%;',
    '  object-fit: contain;',
    '}'
  ].join('')
  var styleDOM = document.createElement('style')
  if (styleDOM.styleSheet) {
    styleDOM.styleSheet.cssText = css
  } else {
    styleDOM.appendChild(document.createTextNode(css))
  }
  document.querySelector('head').appendChild(styleDOM)

  window.addEventListener('resize', toCenter)

  for (let i = 0; i < imgArr.length; i++) {
    imgArr[i].addEventListener('click', imgClickEvent, true)
  }

  function prevent(ev) {
    ev.preventDefault()
  }

  function toCenter() {
    if (!imgMoveOrigin) {
      return
    }
    let availableWidth = document.documentElement.clientWidth
    let width = Math.min(imgMoveOrigin.naturalWidth, parseInt(availableWidth * 0.9))
    let height = (width * imgMoveOrigin.naturalHeight) / imgMoveOrigin.naturalWidth
    if (window.innerHeight * 0.95 < height) {
      height = Math.min(imgMoveOrigin.naturalHeight, parseInt(window.innerHeight * 0.95))
      width = (height * imgMoveOrigin.naturalWidth) / imgMoveOrigin.naturalHeight
    }

    let img = document.querySelector('.img-move-item')
    img.style.left = (availableWidth - width) / 2 + 'px'
    img.style.top = (window.innerHeight - height) / 2 + 'px'
    img.style.width = width + 'px'
    img.style.height = height + 'px'
  }

  function restore() {
    if (restoreLock == true) {
      return
    }
    restoreLock = true
    let div = document.querySelector('.img-move-bg')
    let img = document.querySelector('.img-move-item')

    // 移除键盘事件监听器
    document.removeEventListener('keydown', currentKeyHandler)

    div.style.opacity = 0
    img.style.opacity = 0
    img.style.left = imgMoveOrigin.x + 'px'
    img.style.top = imgMoveOrigin.y + 'px'
    img.style.width = imgMoveOrigin.width + 'px'
    img.style.height = imgMoveOrigin.height + 'px'

    // 恢复目录收起按钮
    const tocToggle = document.querySelector('.toc-sidebar-toggle')
    if (tocToggle) {
      tocToggle.style.zIndex = ''
      tocToggle.style.opacity = ''
      tocToggle.style.pointerEvents = ''
    }

    setTimeout(function () {
      restoreLock = false
      document.body.removeChild(div)
      document.body.removeChild(img)
      imgMoveOrigin = null
    }, 300)
  }

  // 当前的键盘事件处理函数
  let currentKeyHandler = null

  function imgClickEvent(event) {
    imgMoveOrigin = event.target

    let div = document.createElement('div')
    div.className = 'img-move-bg'

    let img = document.createElement('img')
    img.className = 'img-move-item'
    img.src = imgMoveOrigin.src
    img.style.left = imgMoveOrigin.x + 'px'
    img.style.top = imgMoveOrigin.y + 'px'
    img.style.width = imgMoveOrigin.width + 'px'
    img.style.height = imgMoveOrigin.height + 'px'
    
    // 添加 alt 文本作为标题，如果有的话
    if (imgMoveOrigin.alt) {
      img.title = imgMoveOrigin.alt
    }

    // 处理目录收起按钮
    const tocToggle = document.querySelector('.toc-sidebar-toggle')
    if (tocToggle) {
      tocToggle.style.zIndex = '999'
      tocToggle.style.opacity = '0.2'
      tocToggle.style.pointerEvents = 'none'
    }

    div.onclick = function() {
      restore()
    }
    div.onmousewheel = div.onclick
    div.ontouchmove = prevent

    img.onclick = div.onclick
    img.onmousewheel = div.onclick
    img.ontouchmove = prevent
    img.ondragstart = prevent

    document.body.appendChild(div)
    document.body.appendChild(img)

    // 添加键盘 ESC 键关闭功能
    function handleKeyDown(e) {
      if (e.key === 'Escape') {
        restore()
      }
    }
    currentKeyHandler = handleKeyDown
    document.addEventListener('keydown', handleKeyDown)

    setTimeout(function () {
      div.style.opacity = 0.7
      img.style.opacity = 1
      toCenter()
    }, 0)
  }
})

// 切换夜间模式
blog.addLoadEvent(function () {
  const $el = document.querySelector('.footer-btn.theme-toggler')
  const $icon = $el.querySelector('.svg-icon')

  blog.removeClass($el, 'hide')
  if (blog.darkMode) {
    blog.removeClass($icon, 'icon-theme-light')
    blog.addClass($icon, 'icon-theme-dark')
  }

  function initDarkMode(flag) {
    blog.removeClass($icon, 'icon-theme-light')
    blog.removeClass($icon, 'icon-theme-dark')
    if (flag === 'true') blog.addClass($icon, 'icon-theme-dark')
    else blog.addClass($icon, 'icon-theme-light')

    document.documentElement.setAttribute('transition', '')
    setTimeout(function () {
      document.documentElement.removeAttribute('transition')
    }, 600)

    blog.initDarkMode(flag)
  }

  blog.addEvent($el, 'click', function () {
    const flag = blog.darkMode ? 'false' : 'true'
    localStorage.darkMode = flag
    initDarkMode(flag)
  })

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addListener(function (ev) {
      const systemDark = ev.target.matches
      if (systemDark !== blog.darkMode) {
        localStorage.darkMode = '' // 清除用户设置
        initDarkMode(systemDark ? 'true' : 'false')
      }
    })
  }
})

// 标题定位
blog.addLoadEvent(function () {
  if (!document.querySelector('.page-post')) {
    return
  }
  const list = document.querySelectorAll('.post h1, .post h2')
  for (var i = 0; i < list.length; i++) {
    blog.addEvent(list[i], 'click', function (event) {
      const el = event.target
      if (el.scrollIntoView) {
        el.scrollIntoView({ block: 'start' })
      }
      if (el.id && history.replaceState) {
        history.replaceState({}, '', '#' + el.id)
      }
    })
  }
})

// 自动生成文章目录
blog.addLoadEvent(function () {
  // 仅在文章页面执行
  if (!document.querySelector('.page-post')) {
    return
  }

  // 创建目录容器
  const tocContainer = document.createElement('div')
  tocContainer.className = 'toc-container'
  
  // 创建目录标题
  const tocTitle = document.createElement('div')
  tocTitle.className = 'toc-title'
  tocTitle.textContent = '目录'
  
  // 添加展开/折叠图标
  const toggleIcon = document.createElement('span')
  toggleIcon.className = 'toc-toggle-icon'
  toggleIcon.innerHTML = '−' // 默认展开状态显示减号
  tocTitle.appendChild(toggleIcon)
  
  // 添加点击事件，实现折叠/展开功能
  tocTitle.addEventListener('click', function() {
    const tocList = tocContainer.querySelector('.toc-list')
    if (tocList.style.display === 'none') {
      // 展开目录
      tocList.style.display = 'block'
      toggleIcon.innerHTML = '−' // 展开状态显示减号
      localStorage.setItem('tocContentCollapsed', 'false')
    } else {
      // 折叠目录
      tocList.style.display = 'none'
      toggleIcon.innerHTML = '+' // 折叠状态显示加号
      localStorage.setItem('tocContentCollapsed', 'true')
    }
  })
  
  tocContainer.appendChild(tocTitle)

  // 创建目录列表
  const tocList = document.createElement('ul')
  tocList.className = 'toc-list'
  
  // 获取所有标题元素
  const headings = document.querySelectorAll('.page-post h1, .page-post h2, .page-post h3, .page-post h4, .page-post h5, .page-post h6')
  
  // 为每个标题生成目录项
  headings.forEach((heading, index) => {
    // 确保标题有id
    if (!heading.id) {
      heading.id = 'toc-heading-' + index
    }

    const level = parseInt(heading.tagName.charAt(1))
    const listItem = document.createElement('li')
    listItem.className = 'toc-item toc-level-' + level
    
    const link = document.createElement('a')
    link.href = '#' + heading.id
    link.textContent = heading.textContent
    
    // 点击目录项时滚动到对应位置
    link.addEventListener('click', (e) => {
      e.preventDefault()
      heading.scrollIntoView({ behavior: 'smooth', block: 'start' })
      history.pushState(null, null, '#' + heading.id)
    })

    listItem.appendChild(link)
    tocList.appendChild(listItem)
  })

  tocContainer.appendChild(tocList)

  // 创建侧边栏切换按钮
  const sidebarToggle = document.createElement('div')
  sidebarToggle.className = 'toc-sidebar-toggle'
  sidebarToggle.setAttribute('title', '切换目录')
  // 将按钮添加到body中，而不是tocContainer中
  document.body.appendChild(sidebarToggle)

  // 添加脉动动画，使按钮更加明显
  sidebarToggle.classList.add('pulse')
  setTimeout(() => {
    sidebarToggle.classList.remove('pulse')
  }, 6000) // 6秒后移除脉动动画

  // 添加侧边栏切换事件
  sidebarToggle.addEventListener('click', toggleSidebar)
  sidebarToggle.addEventListener('touchend', function(e) {
    e.preventDefault()
    toggleSidebar()
  })

  // 侧边栏切换函数
  function toggleSidebar() {
    tocContainer.classList.toggle('collapsed')
    sidebarToggle.classList.toggle('collapsed')
    if (tocContainer.classList.contains('collapsed')) {
      localStorage.setItem('tocSidebarCollapsed', 'true')
    } else {
      localStorage.setItem('tocSidebarCollapsed', 'false')
    }
  }

  // 将目录插入到 .post 元素之前
  const postContent = document.querySelector('.page-post .post')
  if (postContent) {
    postContent.parentNode.insertBefore(tocContainer, postContent)
    
    // 检查本地存储中的折叠状态并应用
    if (localStorage.getItem('tocContentCollapsed') === 'true') {
      tocList.style.display = 'none'
      toggleIcon.innerHTML = '+'
    }

    // 检查侧边栏折叠状态并应用
    if (localStorage.getItem('tocSidebarCollapsed') === 'true') {
      tocContainer.classList.add('collapsed')
      sidebarToggle.classList.add('collapsed')
    }
    
    // 添加滚动监听，高亮当前可见的标题对应的目录项
    const tocLinks = tocList.querySelectorAll('a')
    const headingElements = []
    
    // 收集所有标题元素及其对应的目录链接
    tocLinks.forEach(link => {
      const targetId = link.getAttribute('href').substring(1)
      const targetHeading = document.getElementById(targetId)
      if (targetHeading) {
        headingElements.push({
          heading: targetHeading,
          link: link
        })
      }
    })
    
    // 滚动时更新活跃的目录项
    function updateActiveHeading() {
      let activeIndex = 0
      const scrollTop = window.scrollY
      
      // 找到当前可见的标题
      for (let i = 0; i < headingElements.length; i++) {
        const headingTop = headingElements[i].heading.getBoundingClientRect().top + scrollTop
        if (headingTop - 100 <= scrollTop) {
          activeIndex = i
        } else {
          break
        }
      }
      
      // 移除所有活跃类
      tocLinks.forEach(link => {
        link.parentNode.classList.remove('active')
      })
      
      // 添加活跃类到当前项
      if (headingElements.length > 0) {
        headingElements[activeIndex].link.parentNode.classList.add('active')
      }
    }
    
    // 初始化活跃标题
    updateActiveHeading()
    
    // 添加滚动监听
    window.addEventListener('scroll', updateActiveHeading)
  }
})

// 确保文本选择功能正常工作
blog.addLoadEvent(function() {
  // 移除可能禁用文本选择的事件监听器
  document.onmousedown = null;
  document.onselectstart = null;
  
  // 确保文章内容可以被选择
  const postContent = document.querySelector('.page-post');
  if (postContent) {
    postContent.style.userSelect = 'text';
    postContent.style.webkitUserSelect = 'text';
    postContent.style.mozUserSelect = 'text';
    postContent.style.msUserSelect = 'text';
    
    // 遍历所有子元素，确保它们也可以被选择
    const allElements = postContent.querySelectorAll('*');
    allElements.forEach(function(el) {
      el.style.userSelect = 'text';
      el.style.webkitUserSelect = 'text';
      el.style.mozUserSelect = 'text';
      el.style.msUserSelect = 'text';
    });
  }
});
