class BatchSearchManager:
    def __init__(self, stack_search, search_list, min_observations):
        """
        Initialize the context manager with an instance of StackSearch,
        the list of trajectories to search, and the minimum number of observations.

        Parameters:
        - stack_search: Instance of the StackSearch class.
        - search_list: List of trajectories to search.
        - min_observations: Minimum number of observations for the search.
        """
        self.stack_search = stack_search
        self.search_list = search_list
        self.min_observations = min_observations

    def __enter__(self):
        """
        This method is called when entering the context managed by the `with` statement.
        It prepares the batch search by calling `prepare_batch_search` on the StackSearch instance.
        """
        # Initialize or prepare memory for the batch search.
        self.stack_search.prepare_batch_search(self.search_list, self.min_observations)
        # Return the object that should be used within the `with` block. Here, it's the StackSearch instance.
        return self.stack_search

    def __exit__(self, exc_type, exc_value, traceback):
        """
        This method is called when exiting the context.
        It cleans up resources by calling `finish_search` on the StackSearch instance.

        Parameters:
        - exc_type: The exception type if an exception was raised in the `with` block.
        - exc_value: The exception value if an exception was raised.
        - traceback: The traceback object if an exception was raised.
        """
        # Clean up resources or delete initialized memory.
        self.stack_search.finish_search()
        # Returning False means any exception raised within the `with` block will be propagated.
        # To suppress exceptions, return True.
        return False
