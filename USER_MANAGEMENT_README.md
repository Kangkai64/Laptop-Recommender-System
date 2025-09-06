# User Management and Behavior Tracking System

This document describes the new user management and behavior tracking features added to the Laptop Recommender System.

## Features

### 1. User Management
- **Create New Users**: Create new user profiles with username and email
- **Select Existing Users**: Choose from users already in the rating dataset
- **User Profiles**: View user information, activity statistics, and preferences
- **User Preferences**: Set and save user preferences for better recommendations

### 2. Behavior Tracking
- **View Tracking**: Automatically track which laptops users view
- **Rating System**: Users can rate and review laptops (1-5 stars with comments)
- **Activity History**: Complete history of user interactions
- **Statistics**: Comprehensive user activity statistics

### 3. Web Interface
- **User Management Page**: Complete interface for user operations
- **Rating Component**: Integrated rating system on laptop detail pages
- **Activity Dashboard**: View user's view history, ratings, and behavior
- **Preferences Management**: Set and update user preferences

## How to Use

### 1. Access User Management
1. Navigate to the "User Management" page from the main navigation
2. You can either create a new user or select an existing user from the rating dataset

### 2. Create a New User
1. Fill in the username (required) and email (optional)
2. Click "Create User"
3. The system will create a new user profile and automatically select it

### 3. Select an Existing User
1. Choose from the dropdown list of existing users
2. Users from the rating dataset will be marked as "Existing user"
3. Click "Select User" to load the user profile

### 4. User Dashboard
Once a user is selected, you can:
- View user statistics (views, ratings, comments)
- See view history of laptops
- View rating and review history
- Set user preferences
- Track complete activity history

### 5. Rate and Review Laptops
1. Navigate to any laptop detail page
2. If you're logged in as a user, you'll see a rating component
3. Select a star rating (1-5)
4. Optionally add a comment
5. Submit your rating

### 6. Automatic View Tracking
- Views are automatically tracked when you visit laptop detail pages
- View counts are incremented for repeat visits
- View history is maintained with timestamps

## Technical Implementation

### Database Structure
The system uses SQLite database with the following tables:
- `users`: User profiles and basic information
- `user_behavior`: Complete behavior tracking
- `user_ratings`: Quick access to user ratings
- `user_views`: Quick access to view history

### API Endpoints
- `GET /api/users` - List all users
- `POST /api/users` - Create new user
- `GET /api/users/<user_id>` - Get user details
- `POST /api/users/<user_id>/ratings` - Submit rating
- `POST /api/users/<user_id>/behavior` - Track behavior
- `GET /api/users/<user_id>/views` - Get view history
- `GET /api/users/<user_id>/ratings` - Get rating history
- `PUT /api/users/<user_id>/preferences` - Update preferences

### Files Added/Modified
- `user_management.py` - Core user management system
- `templates/user_management.html` - User management interface
- `templates/user_rating_component.html` - Rating component
- `app.py` - Added API routes and integration
- `templates/base.html` - Added navigation link
- `templates/laptop_detail.html` - Added rating component and view tracking

## User Experience Flow

1. **First Visit**: User goes to User Management page
2. **User Selection**: User creates new profile or selects existing user
3. **Browsing**: User browses laptops, views are automatically tracked
4. **Rating**: User can rate and review laptops they've viewed
5. **Preferences**: User can set preferences for better recommendations
6. **History**: User can view their complete activity history

## Benefits

### For Users
- **Personalized Experience**: Track preferences and behavior
- **Rating System**: Rate and review laptops
- **Activity History**: See what they've viewed and rated
- **Better Recommendations**: System learns from user behavior

### For System
- **Behavioral Data**: Rich data for improving recommendations
- **User Engagement**: Users can interact with the system
- **Analytics**: Better understanding of user preferences
- **Personalization**: Tailored recommendations based on user history

## Future Enhancements

1. **Recommendation Integration**: Use user behavior for personalized recommendations
2. **Social Features**: Share ratings and reviews
3. **Wishlist**: Save laptops for later viewing
4. **Comparison History**: Track which laptops users compare
5. **Advanced Analytics**: User behavior patterns and insights

## Usage Examples

### Creating a New User
```javascript
// User creates account with username "john_doe"
fetch('/api/users', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        username: 'john_doe',
        email: 'john@example.com'
    })
});
```

### Rating a Laptop
```javascript
// User rates a laptop 4 stars with a comment
fetch('/api/users/user_id/ratings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        laptop_id: 123,
        rating: 4.0,
        comment: 'Great laptop for gaming!'
    })
});
```

### Tracking a View
```javascript
// Automatically track when user views a laptop
fetch('/api/users/user_id/behavior', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        laptop_id: 123,
        behavior_type: 'view',
        data: {page: 'laptop_detail'}
    })
});
```

## Database Location
The user data is stored in `data/user_data.db` (SQLite database).

## Security Considerations
- User data is stored locally in SQLite
- No authentication system (users are identified by session/localStorage)
- Consider adding authentication for production use
- User data is not encrypted (consider encryption for sensitive data)

This system provides a solid foundation for user engagement and behavior tracking, enabling more personalized and effective laptop recommendations.
