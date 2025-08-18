        // Sample student data with attendance status
        const allStudents = [
            { name: 'Student Name 1', status: 'present' },
            { name: 'Student Name 2', status: 'present' },
            { name: 'Student Name 3', status: 'late' },
            { name: 'Student Name 4', status: 'absent' },
            { name: 'Student Name 5', status: 'present' },
            { name: 'Student Name 6', status: 'late' },
            { name: 'Student Name 7', status: 'present' },
            { name: 'Student Name 8', status: 'absent' },
            { name: 'Student Name 9', status: 'present' },
            { name: 'Student Name 10', status: 'late' },
            { name: 'Student Name 11', status: 'present' },
            { name: 'Student Name 12', status: 'absent' },
            { name: 'Student Name 13', status: 'present' },
            { name: 'Student Name 14', status: 'late' },
            { name: 'Student Name 15', status: 'present' },
            { name: 'Student Name 16', status: 'present' },
            { name: 'Student Name 17', status: 'absent' },
            { name: 'Student Name 18', status: 'present' },
            { name: 'Student Name 19', status: 'late' },
            { name: 'Student Name 20', status: 'present' },
            { name: 'Student Name 21', status: 'present' },
            { name: 'Student Name 22', status: 'absent' },
            { name: 'Student Name 23', status: 'present' },
            { name: 'Student Name 24', status: 'late' },
            { name: 'Student Name 25', status: 'present' },
            { name: 'Student Name 26', status: 'present' },
            { name: 'Student Name 27', status: 'absent' },
            { name: 'Student Name 28', status: 'present' },
            { name: 'Student Name 29', status: 'late' },
            { name: 'Student Name 30', status: 'present' },
            { name: 'Student Name 31', status: 'present' },
            { name: 'Student Name 32', status: 'absent' },
            { name: 'Student Name 33', status: 'present' },
            { name: 'Student Name 34', status: 'late' },
            { name: 'Student Name 35', status: 'present' },
            { name: 'Student Name 36', status: 'present' }
        ];

        // Populate initial lists
        const students = {
            present: allStudents.filter(s => s.status === 'present').map(s => s.name),
            late: allStudents.filter(s => s.status === 'late').map(s => s.name),
            absent: allStudents.filter(s => s.status === 'absent').map(s => s.name)
        };

        function createStudentItem(name, index) {
            const studentItem = document.createElement('div');
            studentItem.className = 'student-item';
            
            const avatar = document.createElement('div');
            avatar.className = 'student-avatar';
            avatar.textContent = name.charAt(0).toUpperCase();
            
            const nameDiv = document.createElement('div');
            nameDiv.className = 'student-name';
            nameDiv.textContent = name;
            
            studentItem.appendChild(avatar);
            studentItem.appendChild(nameDiv);
            
            return studentItem;
        }

        function populateStudentList(containerId, studentList) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            studentList.forEach((name, index) => {
                const studentItem = createStudentItem(name, index);
                container.appendChild(studentItem);
            });
        }

        function createModalStudentItem(student) {
            const studentItem = document.createElement('div');
            studentItem.className = 'modal-student-item';
            
            const studentInfo = document.createElement('div');
            studentInfo.className = 'modal-student-info';
            
            const avatar = document.createElement('div');
            avatar.className = 'modal-student-avatar';
            avatar.textContent = student.name.charAt(0).toUpperCase();
            
            const nameDiv = document.createElement('div');
            nameDiv.className = 'modal-student-name';
            nameDiv.textContent = student.name;
            
            studentInfo.appendChild(avatar);
            studentInfo.appendChild(nameDiv);
            
            const checkboxes = document.createElement('div');
            checkboxes.className = 'attendance-checkboxes';
            
            // Present checkbox
            const presentWrapper = document.createElement('div');
            presentWrapper.className = 'checkbox-wrapper';
            const presentCheckbox = document.createElement('div');
            presentCheckbox.className = `attendance-checkbox checkbox-present ${student.status === 'present' ? 'checked' : ''}`;
            presentCheckbox.dataset.status = 'present';
            presentCheckbox.dataset.student = student.name;
            presentWrapper.appendChild(presentCheckbox);
            
            // Late checkbox
            const lateWrapper = document.createElement('div');
            lateWrapper.className = 'checkbox-wrapper';
            const lateCheckbox = document.createElement('div');
            lateCheckbox.className = `attendance-checkbox checkbox-late ${student.status === 'late' ? 'checked' : ''}`;
            lateCheckbox.dataset.status = 'late';
            lateCheckbox.dataset.student = student.name;
            lateWrapper.appendChild(lateCheckbox);
            
            // Absent checkbox
            const absentWrapper = document.createElement('div');
            absentWrapper.className = 'checkbox-wrapper';
            const absentCheckbox = document.createElement('div');
            absentCheckbox.className = `attendance-checkbox checkbox-absent ${student.status === 'absent' ? 'checked' : ''}`;
            absentCheckbox.dataset.status = 'absent';
            absentCheckbox.dataset.student = student.name;
            absentWrapper.appendChild(absentCheckbox);
            
            checkboxes.appendChild(presentWrapper);
            checkboxes.appendChild(lateWrapper);
            checkboxes.appendChild(absentWrapper);
            
            studentItem.appendChild(studentInfo);
            studentItem.appendChild(checkboxes);
            
            return studentItem;
        }

        function populateModalStudentsList() {
            const container = document.getElementById('modalStudentsList');
            container.innerHTML = '';
            
            allStudents.forEach(student => {
                const studentItem = createModalStudentItem(student);
                container.appendChild(studentItem);
            });
        }

        // Initialize student lists
        populateStudentList('presentList', students.present);
        populateStudentList('lateList', students.late);
        populateStudentList('absentList', students.absent);
        populateModalStudentsList();

        // Modal functionality
        const editModal = document.getElementById('editModal');
        const editBtn = document.getElementById('editBtn');
        const closeModal = document.getElementById('closeModal');
        const modalCancel = document.getElementById('modalCancel');
        const modalSave = document.getElementById('modalSave');
        const studentSearch = document.getElementById('studentSearch');

        editBtn.addEventListener('click', () => {
            editModal.style.display = 'flex';
        });

        closeModal.addEventListener('click', () => {
            editModal.style.display = 'none';
        });

        modalCancel.addEventListener('click', () => {
            editModal.style.display = 'none';
        });

        editModal.addEventListener('click', (e) => {
            if (e.target === editModal) {
                editModal.style.display = 'none';
            }
        });

        // Checkbox functionality
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('attendance-checkbox')) {
                const studentName = e.target.dataset.student;
                const newStatus = e.target.dataset.status;
                
                // Update student status
                const student = allStudents.find(s => s.name === studentName);
                if (student) {
                    student.status = newStatus;
                }
                
                // Update all checkboxes for this student
                const studentCheckboxes = document.querySelectorAll(`[data-student="${studentName}"]`);
                studentCheckboxes.forEach(cb => {
                    cb.classList.remove('checked');
                    if (cb.dataset.status === newStatus) {
                        cb.classList.add('checked');
                    }
                });
            }
        });

        // Search functionality
        studentSearch.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const studentItems = document.querySelectorAll('.modal-student-item');
            
            studentItems.forEach(item => {
                const studentName = item.querySelector('.modal-student-name').textContent.toLowerCase();
                if (studentName.includes(searchTerm)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        });

        // Bulk actions
        document.querySelectorAll('.attendance-option').forEach(btn => {
            btn.addEventListener('click', () => {
                const status = btn.classList.contains('option-present') ? 'present' :
                              btn.classList.contains('option-late') ? 'late' : 'absent';
                
                // Get visible students (in case search is active)
                const visibleStudents = document.querySelectorAll('.modal-student-item:not([style*="display: none"])');
                
                visibleStudents.forEach(studentItem => {
                    const studentName = studentItem.querySelector('.modal-student-name').textContent;
                    const student = allStudents.find(s => s.name === studentName);
                    if (student) {
                        student.status = status;
                    }
                    
                    // Update checkboxes
                    const checkboxes = studentItem.querySelectorAll('.attendance-checkbox');
                    checkboxes.forEach(cb => {
                        cb.classList.remove('checked');
                        if (cb.dataset.status === status) {
                            cb.classList.add('checked');
                        }
                    });
                });
            });
        });

        modalSave.addEventListener('click', () => {
            // Update the main lists based on new statuses
            const updatedStudents = {
                present: allStudents.filter(s => s.status === 'present').map(s => s.name),
                late: allStudents.filter(s => s.status === 'late').map(s => s.name),
                absent: allStudents.filter(s => s.status === 'absent').map(s => s.name)
            };
            
            // Update counts
            document.querySelector('.count-present').textContent = `${updatedStudents.present.length}/36`;
            document.querySelector('.count-late').textContent = `${updatedStudents.late.length}/36`;
            document.querySelector('.count-absent').textContent = `${updatedStudents.absent.length}/36`;
            
            // Repopulate lists
            populateStudentList('presentList', updatedStudents.present);
            populateStudentList('lateList', updatedStudents.late);
            populateStudentList('absentList', updatedStudents.absent);
            
            // Close modal
            editModal.style.display = 'none';
            
            alert('Attendance updated successfully!');
        });

        // Add button functionality
        document.querySelector('.btn-save').addEventListener('click', function() {
            alert('Attendance summary saved!');
        });

        document.querySelector('.btn-cancel').addEventListener('click', function() {
            if (confirm('Are you sure you want to cancel? Unsaved changes will be lost.')) {
                console.log('Cancelled');
            }
        });

        // Update time every second (for demonstration)
        function updateCurrentTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
            }).toLowerCase();
        }

        setInterval(updateCurrentTime, 1000);