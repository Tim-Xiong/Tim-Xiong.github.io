# My Portfolio Site

This is a personal portfolio site built with [Zola](https://www.getzola.org/). Here is a [demo video](https://youtu.be/kfvQn4bmGkw).

## Structure

- **Content:** Holds the main site content, including future blog posts and project details.
- **Themes/DeepThought:** The selected theme, including styles, templates, and static assets.
- **config.toml:** Configuration file for Zola.

## Setup & Usage

1. Install [Zola](https://www.getzola.org/documentation/getting-started/installation/)
2. Clone this repository:
   ```sh
   git clone <repo-url>
   cd <repo-name>
   ```

3. Local Development:
   ```sh
   # Start the Zola development server
   zola serve
   ```
   This will start a local server at `http://127.0.0.1:1111`. The site will auto-reload when you make changes.

4. Creating Content:
   ```sh
   # Create a new blog post
   touch content/posts/my-new-post.md
   ```
   Add front matter at the top of your post:
   ```md
   +++
   title = "My New Post"
   date = 2024-01-31
   description = "Description of your post"
   +++

   Your content here...
   ```

5. Building the Site:
   ```sh
   # Generate the static site
   zola build
   ```
   The output will be in the `public` directory.

6. Customization:
   - Edit `config.toml` to modify site settings
   - Update theme settings in `themes/DeepThought/config.toml`
   - Modify templates in `themes/DeepThought/templates/`
   - Add custom CSS in `themes/DeepThought/sass/`
