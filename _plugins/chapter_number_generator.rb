module Jekyll
  class ChapterNumberGenerator < Generator
    safe true
    priority :high

    def generate(site)
      # 按书籍ID分组
      books = {}
      
      site.collections['books'].docs.each do |doc|
        book_id = doc.data['book_id']
        next unless book_id
        
        books[book_id] ||= []
        books[book_id] << doc
      end
      
      # 为每本书的章节生成编号
      books.each do |book_id, chapters|
        # 从文件名中提取章节编号
        chapters_with_number = []
        
        chapters.each do |chapter|
          file_path = chapter.path
          file_name = File.basename(file_path, '.*')
          
          # 尝试从文件名中提取数字部分
          if file_name =~ /ch(\d+)(?:[_\.](\d+))?/i
            main_number = $1.to_i
            sub_number = $2 ? $2.to_i : 0
            # 使用浮点数进行排序，主章节号.子章节号
            sort_number = main_number + (sub_number / 100.0)
            chapters_with_number << [chapter, sort_number, "#{main_number}#{sub_number > 0 ? '.'+sub_number.to_s : ''}"]
          else
            # 如果无法提取，放在最后
            chapters_with_number << [chapter, 999, nil]
          end
        end
        
        # 按照提取的章节编号排序
        sorted_chapters = chapters_with_number.sort_by { |_, sort_num, _| sort_num }.map { |ch, _, _| ch }
        
        # 为排序后的章节设置display_chapter_number
        sorted_chapters.each_with_index do |chapter, index|
          # 首先尝试从文件名中提取显示编号
          file_name = File.basename(chapter.path, '.*')
          
          if file_name =~ /ch(\d+)(?:[_\.](\d+))?/i
            main_number = $1
            sub_number = $2
            if sub_number
              display_number = "#{main_number}.#{sub_number}"
            else
              display_number = main_number
            end
          else
            # 如果无法从文件名提取，使用索引+1
            display_number = (index + 1).to_s
          end
          
          # 设置显示编号
          chapter.data['display_chapter_number'] = display_number
          
          # 设置上一章和下一章的链接
          if index > 0
            chapter.data['previous_chapter'] = sorted_chapters[index - 1].url
          end
          
          if index < sorted_chapters.size - 1
            chapter.data['next_chapter'] = sorted_chapters[index + 1].url
          end
        end
      end
    end
  end
end 