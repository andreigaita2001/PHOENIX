#!/usr/bin/env python3
"""
Tennis Scheduling Module - Manages tennis coaching schedules.
Designed specifically for a tennis coach working at Tennis 13.
"""

import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import calendar
from enum import Enum

class LessonType(Enum):
    PERMANENT = "permanent"
    ONE_TIME = "one_time"
    RECURRING = "recurring"

class TennisScheduler:
    """
    Manages tennis lesson scheduling with visual representations.
    """

    def __init__(self, memory_manager=None):
        """
        Initialize the Tennis Scheduler.

        Args:
            memory_manager: Memory system for persistence
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.TennisScheduler")

        # Schedule storage
        self.data_dir = Path('./data/schedules')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.schedule_file = self.data_dir / 'tennis_schedule.json'

        # Coach information
        self.coach_info = {
            'name': 'Tennis Coach',
            'workplace': 'Tennis 13',
            'work_start': '14:00',  # 2 PM
            'travel_time': 80,  # 1h20min in minutes
            'notification_lead': 90  # Notification time before lesson
        }

        # Load existing schedule
        self.lessons = self._load_schedule()

        self.logger.info("Tennis Scheduler initialized")

    def _load_schedule(self) -> Dict[str, List[Dict]]:
        """Load schedule from disk."""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load schedule: {e}")

        # Return default structure
        return {
            'permanent': [],
            'scheduled': [],
            'pending_court_booking': []
        }

    def _save_schedule(self):
        """Save schedule to disk."""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.lessons, f, indent=2, default=str)

            # Also save to memory if available
            if self.memory_manager:
                self.memory_manager.learn_fact(
                    f"Schedule updated: {len(self.lessons['scheduled'])} lessons scheduled",
                    category='scheduling'
                )
        except Exception as e:
            self.logger.error(f"Failed to save schedule: {e}")

    def add_permanent_lesson(self, day_of_week: str, start_time: str,
                           end_time: str, student_name: str = "Regular Student") -> bool:
        """
        Add a permanent weekly lesson.

        Args:
            day_of_week: Day of the week (e.g., 'Monday')
            start_time: Start time (HH:MM format)
            end_time: End time (HH:MM format)
            student_name: Name of the student

        Returns:
            Success status
        """
        try:
            lesson = {
                'type': LessonType.PERMANENT.value,
                'day_of_week': day_of_week,
                'start_time': start_time,
                'end_time': end_time,
                'student_name': student_name,
                'created': datetime.now().isoformat(),
                'court_booked': True  # Permanent lessons have courts pre-booked
            }

            self.lessons['permanent'].append(lesson)
            self._save_schedule()

            self.logger.info(f"Added permanent lesson: {day_of_week} {start_time}-{end_time}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add permanent lesson: {e}")
            return False

    def add_lesson_from_natural_language(self, description: str) -> Dict[str, Any]:
        """
        Add a lesson from natural language description.

        Args:
            description: Natural language description of the lesson

        Returns:
            Result of the operation
        """
        # Parse the description
        import re

        result = {
            'success': False,
            'message': '',
            'lesson': None,
            'needs_court_booking': False
        }

        # Try to extract date/time information
        # Look for patterns like "tomorrow at 3pm", "Monday 2-3pm", etc.

        # Simple pattern matching (can be enhanced)
        time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
        day_pattern = r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)'

        day_match = re.search(day_pattern, description.lower())
        time_matches = re.findall(time_pattern, description.lower())

        if not day_match or not time_matches:
            result['message'] = "Could not parse date/time from description"
            return result

        # Convert to actual date
        if day_match.group(1) == 'today':
            lesson_date = datetime.now().date()
        elif day_match.group(1) == 'tomorrow':
            lesson_date = (datetime.now() + timedelta(days=1)).date()
        else:
            # Find next occurrence of this day
            target_day = day_match.group(1).capitalize()
            today = datetime.now().date()
            days_ahead = 0
            for i in range(7):
                if calendar.day_name[(today + timedelta(days=i)).weekday()] == target_day:
                    days_ahead = i
                    break
            lesson_date = today + timedelta(days=days_ahead)

        # Parse times
        if len(time_matches) >= 1:
            start_hour = int(time_matches[0][0])
            if time_matches[0][2] == 'pm' and start_hour != 12:
                start_hour += 12
            start_time = f"{start_hour:02d}:{time_matches[0][1] or '00'}"

            # Default to 1 hour lesson if end time not specified
            if len(time_matches) >= 2:
                end_hour = int(time_matches[1][0])
                if time_matches[1][2] == 'pm' and end_hour != 12:
                    end_hour += 12
                end_time = f"{end_hour:02d}:{time_matches[1][1] or '00'}"
            else:
                end_time = f"{(start_hour + 1) % 24:02d}:{time_matches[0][1] or '00'}"

            # Extract student name if mentioned
            name_pattern = r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            name_match = re.search(name_pattern, description)
            student_name = name_match.group(1) if name_match else "New Student"

            # Add the lesson
            lesson = {
                'type': LessonType.ONE_TIME.value,
                'date': lesson_date.isoformat(),
                'start_time': start_time,
                'end_time': end_time,
                'student_name': student_name,
                'created': datetime.now().isoformat(),
                'court_booked': False,
                'original_request': description
            }

            self.lessons['scheduled'].append(lesson)
            self.lessons['pending_court_booking'].append(lesson)
            self._save_schedule()

            result['success'] = True
            result['lesson'] = lesson
            result['needs_court_booking'] = True
            result['message'] = f"Added lesson on {lesson_date} from {start_time} to {end_time} with {student_name}"

        return result

    def get_week_schedule(self, week_offset: int = 0) -> Dict[str, List[Dict]]:
        """
        Get schedule for a specific week.

        Args:
            week_offset: Number of weeks from current week

        Returns:
            Schedule organized by day
        """
        # Get the start of the target week
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)

        week_schedule = {}

        for i in range(7):
            current_date = start_of_week + timedelta(days=i)
            day_name = calendar.day_name[current_date.weekday()]

            day_lessons = []

            # Add permanent lessons for this day
            for lesson in self.lessons['permanent']:
                if lesson['day_of_week'] == day_name:
                    day_lessons.append({
                        **lesson,
                        'date': current_date.isoformat()
                    })

            # Add scheduled lessons for this date
            for lesson in self.lessons['scheduled']:
                if lesson.get('date') == current_date.isoformat():
                    day_lessons.append(lesson)

            # Sort by start time
            day_lessons.sort(key=lambda x: x['start_time'])
            week_schedule[day_name] = day_lessons

        return week_schedule

    def get_visual_schedule(self, week_offset: int = 0) -> str:
        """
        Get a visual text representation of the schedule.

        Args:
            week_offset: Number of weeks from current week

        Returns:
            Formatted schedule string
        """
        week = self.get_week_schedule(week_offset)

        # Calculate week dates
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
        end_of_week = start_of_week + timedelta(days=6)

        output = []
        output.append("=" * 60)
        output.append(f"📅 TENNIS SCHEDULE - Week of {start_of_week.strftime('%B %d, %Y')}")
        output.append("=" * 60)

        for day_name in calendar.day_name:
            day_date = start_of_week + timedelta(days=list(calendar.day_name).index(day_name))
            output.append(f"\n📆 {day_name}, {day_date.strftime('%B %d')}")
            output.append("-" * 40)

            if week[day_name]:
                for lesson in week[day_name]:
                    status = "✅" if lesson.get('court_booked', False) else "⚠️ Court pending"
                    lesson_type = "🔄 Permanent" if lesson['type'] == 'permanent' else "📍 Scheduled"

                    output.append(f"  {lesson['start_time']} - {lesson['end_time']}")
                    output.append(f"    Student: {lesson['student_name']}")
                    output.append(f"    Status: {status} | Type: {lesson_type}")
            else:
                output.append("  No lessons scheduled")

        # Add reminders section
        output.append("\n" + "=" * 60)
        output.append("🔔 REMINDERS")
        output.append("-" * 40)

        # Check for lessons needing court booking
        if self.lessons['pending_court_booking']:
            output.append(f"⚠️ {len(self.lessons['pending_court_booking'])} lessons need court booking at Tennis 13!")
            for lesson in self.lessons['pending_court_booking'][:3]:
                output.append(f"   - {lesson['date']} at {lesson['start_time']}")

        # Check for upcoming lessons today
        today_name = calendar.day_name[datetime.now().weekday()]
        if week[today_name]:
            now = datetime.now().time()
            for lesson in week[today_name]:
                lesson_time = datetime.strptime(lesson['start_time'], '%H:%M').time()
                time_diff = datetime.combine(datetime.today(), lesson_time) - datetime.now()

                if timedelta() < time_diff < timedelta(hours=2):
                    output.append(f"🎾 Lesson with {lesson['student_name']} in {int(time_diff.total_seconds()/60)} minutes!")

        return "\n".join(output)

    def mark_court_booked(self, lesson_identifier: str) -> bool:
        """
        Mark a lesson's court as booked.

        Args:
            lesson_identifier: Date and time or other identifier

        Returns:
            Success status
        """
        for lesson in self.lessons['pending_court_booking']:
            if (lesson.get('date') in lesson_identifier and
                lesson.get('start_time') in lesson_identifier):
                lesson['court_booked'] = True
                self.lessons['pending_court_booking'].remove(lesson)
                self._save_schedule()
                return True
        return False

    def get_next_lesson(self) -> Optional[Dict]:
        """Get the next upcoming lesson."""
        now = datetime.now()
        next_lesson = None
        min_time_diff = timedelta(days=365)

        # Check all scheduled lessons
        all_lessons = []

        # Add today's permanent lessons
        today_name = calendar.day_name[now.weekday()]
        for lesson in self.lessons['permanent']:
            if lesson['day_of_week'] == today_name:
                lesson_datetime = datetime.combine(
                    now.date(),
                    datetime.strptime(lesson['start_time'], '%H:%M').time()
                )
                if lesson_datetime > now:
                    all_lessons.append((lesson_datetime, lesson))

        # Add scheduled lessons
        for lesson in self.lessons['scheduled']:
            lesson_date = datetime.fromisoformat(lesson['date'])
            lesson_datetime = datetime.combine(
                lesson_date,
                datetime.strptime(lesson['start_time'], '%H:%M').time()
            )
            if lesson_datetime > now:
                all_lessons.append((lesson_datetime, lesson))

        # Find the earliest
        if all_lessons:
            all_lessons.sort(key=lambda x: x[0])
            return all_lessons[0][1]

        return None

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return actual capabilities of this module.

        Returns:
            Dictionary of capabilities
        """
        return {
            'add_lessons': True,
            'natural_language_input': True,
            'visual_schedule': True,
            'permanent_lessons': True,
            'court_booking_tracking': True,
            'reminders': True,
            'travel_time_calculation': True,
            'export_calendar': False,  # Not yet implemented
            'sync_google_calendar': False,  # Not yet implemented
            'send_notifications': False  # Not yet implemented
        }