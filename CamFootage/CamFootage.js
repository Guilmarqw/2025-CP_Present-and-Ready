document.addEventListener('DOMContentLoaded', () => {
    // Update datetime
    function updateDateTime() {
        const now = new Date();
        const formatted = now.toLocaleString('en-US', {
            month: '2-digit',
            day: '2-digit',
            year: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).replace(',', '-');
        document.getElementById('date-time').textContent = formatted;
    }
    
    updateDateTime();
    setInterval(updateDateTime, 60000);

    // Sample student data
    const students = [
        'Student Name 1',
        'Student Name 2',
        'Student Name 3',
        'Student Name 4',
        'Student Name 5',
        'Student Name 6',
        'Student Name 7',
        'Student Name 8',
    ];

    // Populate student list
    const studentList = document.querySelector('.student-list');
    students.forEach(name => {
        const div = document.createElement('div');
        div.className = 'student-row';
        div.innerHTML = `
            <img src="avatar-placeholder.png" alt="Student" class="avatar">
            <span class="student-name">${name}</span>
            <span class="status present">Present</span>
        `;
        studentList.appendChild(div);
    });
});
